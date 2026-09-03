use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use spirv_cross2::reflect::{BitWidth, ResourceType, Scalar, ScalarKind, StructMember, TypeInner};
use spirv_cross2::spirv::StorageClass;
use spirv_cross2::targets;
use spirv_cross2::{Compiler, Module};

#[derive(Debug, Clone)]
struct Job {
    spv_path: PathBuf,
    struct_name: String,
}

#[derive(Debug, Parser)]
#[command(name = "reflect_push_constants")]
#[command(about = "Generate C++ push-constant structs from SPIR-V")]
struct Options {
    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    preamble: PathBuf,

    #[arg(
        long = "shader",
        value_names = ["STRUCT", "SPV"],
        num_args = 2,
        required = true
    )]
    shaders: Vec<String>,
}

#[derive(Debug)]
struct Field {
    name: String,
    cpp_type: &'static str,
    offset: u32,
    size: u32,
}

#[derive(Debug)]
struct ReflectedStruct {
    job: Job,
    size: u32,
    fields: Vec<Field>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("reflect_push_constants: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let options = Options::parse();
    let jobs = parse_jobs(&options.shaders)?;
    validate_struct_names(&jobs)?;

    let preamble = fs::read_to_string(&options.preamble)
        .with_context(|| format!("cannot read {}", options.preamble.display()))?;

    let reflected = jobs.iter().map(reflect_job).collect::<Result<Vec<_>>>()?;

    let output = generate_header(&preamble, &reflected);
    let written = write_if_changed(&options.output, &output)?;

    if written {
        println!(
            "Generated {} from {} shader{}",
            options.output.display(),
            jobs.len(),
            if jobs.len() == 1 { "" } else { "s" }
        );
    }

    Ok(())
}

fn parse_jobs(values: &[String]) -> Result<Vec<Job>> {
    let chunks = values.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        bail!("--shader requires <StructName> <input.spv>");
    }

    Ok(chunks
        .map(|pair| Job {
            struct_name: pair[0].clone(),
            spv_path: PathBuf::from(&pair[1]),
        })
        .collect())
}

fn validate_struct_names(jobs: &[Job]) -> Result<()> {
    let mut names = HashSet::with_capacity(jobs.len());

    for job in jobs {
        if !names.insert(job.struct_name.as_str()) {
            bail!("duplicate generated struct name: {}", job.struct_name);
        }
    }

    Ok(())
}

fn reflect_job(job: &Job) -> Result<ReflectedStruct> {
    let words = read_spirv_words(&job.spv_path)?;
    let compiler = Compiler::<targets::None>::new(Module::from_words(&words))
        .with_context(|| format!("{}: cannot parse SPIR-V", job.spv_path.display()))?;

    let resources = compiler
        .shader_resources()
        .with_context(|| format!("{}: cannot enumerate shader resources", job.spv_path.display()))?;
    let push_constants = resources
        .resources_for_type(ResourceType::PushConstant)
        .with_context(|| format!("{}: cannot enumerate push constants", job.spv_path.display()))?
        .collect::<Vec<_>>();

    if push_constants.len() != 1 {
        bail!(
            "{}: expected exactly one push-constant block, found {}",
            job.spv_path.display(),
            push_constants.len()
        );
    }

    let block_type = compiler
        .type_description(push_constants[0].base_type_id)
        .with_context(|| format!("{}: cannot inspect push-constant type", job.spv_path.display()))?;

    let TypeInner::Struct(block) = block_type.inner else {
        bail!("{}: push-constant resource is not a struct", job.spv_path.display());
    };

    let mut fields = Vec::with_capacity(block.members.len() * 2);
    for member in &block.members {
        flatten_member(&compiler, member, 0, &mut fields)
            .with_context(|| job.spv_path.display().to_string())?;
    }

    fields.sort_by_key(|field| field.offset);
    validate_fields(&fields, block.size, &job.spv_path)?;

    Ok(ReflectedStruct {
        job: job.clone(),
        size: u32::try_from(block.size).context("push-constant block is larger than 4 GiB")?,
        fields,
    })
}

fn flatten_member(
    compiler: &Compiler<targets::None>,
    member: &StructMember<'_>,
    base_offset: u32,
    fields: &mut Vec<Field>,
) -> Result<()> {
    let offset = base_offset
        .checked_add(member.offset)
        .context("push-constant member offset overflow")?;
    let ty = compiler.type_description(member.id)?;

    match ty.inner {
        TypeInner::Pointer {
            storage: StorageClass::PhysicalStorageBuffer,
            ..
        } => {
            let mut name = make_identifier(member_name(member))?;
            name.push_str("_address");

            fields.push(Field {
                name,
                cpp_type: "VkDeviceAddress",
                offset,
                size: u32::try_from(member.size).context("pointer member is too large")?,
            });
        }

        TypeInner::Struct(nested) => {
            for nested_member in &nested.members {
                flatten_member(compiler, nested_member, offset, fields)?;
            }
        }

        TypeInner::Array { .. } => {
            bail!(
                "arrays are not currently supported for push-constant member {}",
                display_member_name(member)
            );
        }

        TypeInner::Matrix { .. } => {
            bail!(
                "matrices are not currently supported for push-constant member {}",
                display_member_name(member)
            );
        }

        TypeInner::Scalar(scalar) => {
            let name = make_identifier(member_name(member))?;
            let cpp_type = scalar_cpp_type(&scalar)
                .with_context(|| format!("{}", display_member_name(member)))?;

            fields.push(Field {
                name,
                cpp_type,
                offset,
                size: u32::try_from(scalar.size.byte_size()).unwrap(),
            });
        }

        TypeInner::Vector { width, scalar } => {
            let name = make_identifier(member_name(member))?;
            let cpp_type = scalar_cpp_type(&scalar).with_context(|| name.clone())?;

            if !(2..=4).contains(&width) {
                bail!("{name}: unsupported vector component count {width}");
            }

            let component_size = u32::try_from(scalar.size.byte_size()).unwrap();
            const SUFFIXES: [&str; 4] = ["x", "y", "z", "w"];

            for index in 0..width {
                fields.push(Field {
                    name: format!("{name}_{}", SUFFIXES[index as usize]),
                    cpp_type,
                    offset: offset + index * component_size,
                    size: component_size,
                });
            }
        }

        TypeInner::Pointer { storage, .. } => {
            bail!(
                "unsupported pointer storage class {storage:?} for push-constant member {}",
                display_member_name(member)
            );
        }

        other => {
            bail!(
                "unsupported SPIR-V type {other:?} for push-constant member {}",
                display_member_name(member)
            );
        }
    }

    Ok(())
}

fn scalar_cpp_type(scalar: &Scalar) -> Result<&'static str> {
    match (scalar.kind, scalar.size) {
        (ScalarKind::Float, BitWidth::Word) => Ok("float"),
        (ScalarKind::Float, BitWidth::DoubleWord) => Ok("double"),

        (ScalarKind::Int, BitWidth::Byte) => Ok("std::int8_t"),
        (ScalarKind::Int, BitWidth::HalfWord) => Ok("std::int16_t"),
        (ScalarKind::Int, BitWidth::Word) => Ok("std::int32_t"),
        (ScalarKind::Int, BitWidth::DoubleWord) => Ok("std::int64_t"),

        (ScalarKind::Uint, BitWidth::Byte) => Ok("std::uint8_t"),
        (ScalarKind::Uint, BitWidth::HalfWord) => Ok("std::uint16_t"),
        (ScalarKind::Uint, BitWidth::Word) => Ok("std::uint32_t"),
        (ScalarKind::Uint, BitWidth::DoubleWord) => Ok("std::uint64_t"),

        (ScalarKind::Bool, _) => {
            bail!("bool is not supported; use an explicitly sized integer instead")
        }
        (ScalarKind::Float, width) => bail!("unsupported floating-point width: {width:?}"),
        (kind, width) => bail!("unsupported scalar type: {kind:?} {width:?}"),
    }
}

fn validate_fields(fields: &[Field], block_size: usize, path: &Path) -> Result<()> {
    let mut names = HashSet::with_capacity(fields.len());
    let mut cursor = 0u32;

    for field in fields {
        if !names.insert(field.name.as_str()) {
            bail!(
                "{}: generated duplicate C++ field name '{}'",
                path.display(),
                field.name
            );
        }

        if field.offset < cursor {
            bail!(
                "{}: overlapping reflected field '{}'",
                path.display(),
                field.name
            );
        }

        cursor = field
            .offset
            .checked_add(field.size)
            .context("reflected field extent overflow")?;
    }

    if usize::try_from(cursor).unwrap() > block_size {
        bail!(
            "{}: reflected members extend beyond the push-constant block",
            path.display()
        );
    }

    Ok(())
}

fn generate_header(preamble: &str, reflected: &[ReflectedStruct]) -> String {
    let mut output = String::with_capacity(preamble.len() + reflected.len() * 1024);
    output.push_str(preamble);

    if !output.is_empty() && !output.ends_with('\n') {
        output.push('\n');
    }

    output.push_str(
        "\n// -----------------------------------------------------------------------------\n\
         // Generated shader push constants. Do not edit.\n\
         // -----------------------------------------------------------------------------\n\n",
    );

    for entry in reflected {
        append_struct(&mut output, entry);
    }

    output
}

fn append_struct(output: &mut String, reflected: &ReflectedStruct) {
    writeln!(output, "// Generated from {}", reflected.job.spv_path.file_name().unwrap_or_default().to_string_lossy()).unwrap();
    writeln!(output, "struct {} {{", reflected.job.struct_name).unwrap();

    let mut cursor = 0u32;
    let mut padding_index = 0u32;

    for field in &reflected.fields {
        if field.offset > cursor {
            let padding_size = field.offset - cursor;
            writeln!(
                output,
                "    std::uint8_t generated_padding_{padding_index}[{padding_size}]{{}};"
            )
            .unwrap();
            padding_index += 1;
        }

        writeln!(output, "    {} {}{{}};", field.cpp_type, field.name).unwrap();
        cursor = field.offset + field.size;
    }

    if reflected.size > cursor {
        let padding_size = reflected.size - cursor;
        writeln!(
            output,
            "    std::uint8_t generated_padding_{padding_index}[{padding_size}]{{}};"
        )
        .unwrap();
    }

    writeln!(output, "}};\n").unwrap();
    writeln!(
        output,
        "static_assert(std::is_standard_layout_v<{}>);",
        reflected.job.struct_name
    )
    .unwrap();
    writeln!(
        output,
        "static_assert(sizeof({}) == {});",
        reflected.job.struct_name, reflected.size
    )
    .unwrap();

    for field in &reflected.fields {
        writeln!(
            output,
            "static_assert(offsetof({}, {}) == {});",
            reflected.job.struct_name, field.name, field.offset
        )
        .unwrap();
    }

    output.push('\n');
}

fn read_spirv_words(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;

    if bytes.len() % 4 != 0 {
        bail!(
            "{}: SPIR-V byte length {} is not divisible by four",
            path.display(),
            bytes.len()
        );
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn write_if_changed(path: &Path, contents: &str) -> Result<bool> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    }

    match fs::read_to_string(path) {
        Ok(existing) if existing == contents => return Ok(false),
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("cannot read {}", path.display())),
    }

    fs::write(path, contents).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(true)
}

fn member_name<'a>(member: &'a StructMember<'a>) -> Option<&'a str> {
    member.name.as_ref().map(|name| name.as_ref())
}

fn display_member_name<'a>(member: &'a StructMember<'a>) -> &'a str {
    member_name(member).unwrap_or("<unnamed>")
}

fn make_identifier(name: Option<&str>) -> Result<String> {
    let Some(name) = name.filter(|name| !name.is_empty()) else {
        bail!("encountered unnamed push-constant member");
    };

    let mut result = to_snake_case(name);
    result = result
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();

    if result.as_bytes().first().is_some_and(u8::is_ascii_digit) {
        result.insert(0, '_');
    }

    if is_cpp_keyword(&result) {
        result.push('_');
    }

    Ok(result)
}

fn to_snake_case(name: &str) -> String {
    let bytes = name.as_bytes();
    let mut result = String::with_capacity(name.len() + 4);

    for (index, &byte) in bytes.iter().enumerate() {
        let is_upper = byte.is_ascii_uppercase();
        let previous_is_lower_or_digit = index > 0
            && (bytes[index - 1].is_ascii_lowercase() || bytes[index - 1].is_ascii_digit());
        let next_is_lower = bytes
            .get(index + 1)
            .is_some_and(u8::is_ascii_lowercase);

        if is_upper
            && index != 0
            && bytes[index - 1] != b'_'
            && (previous_is_lower_or_digit || next_is_lower)
        {
            result.push('_');
        }

        result.push((byte as char).to_ascii_lowercase());
    }

    result
}

fn is_cpp_keyword(value: &str) -> bool {
    matches!(
        value,
        "alignas"
            | "alignof"
            | "and"
            | "and_eq"
            | "asm"
            | "auto"
            | "bitand"
            | "bitor"
            | "bool"
            | "break"
            | "case"
            | "catch"
            | "char"
            | "char8_t"
            | "char16_t"
            | "char32_t"
            | "class"
            | "compl"
            | "concept"
            | "const"
            | "consteval"
            | "constexpr"
            | "constinit"
            | "const_cast"
            | "continue"
            | "co_await"
            | "co_return"
            | "co_yield"
            | "decltype"
            | "default"
            | "delete"
            | "do"
            | "double"
            | "dynamic_cast"
            | "else"
            | "enum"
            | "explicit"
            | "export"
            | "extern"
            | "false"
            | "float"
            | "for"
            | "friend"
            | "goto"
            | "if"
            | "inline"
            | "int"
            | "long"
            | "mutable"
            | "namespace"
            | "new"
            | "noexcept"
            | "not"
            | "not_eq"
            | "nullptr"
            | "operator"
            | "or"
            | "or_eq"
            | "private"
            | "protected"
            | "public"
            | "register"
            | "reinterpret_cast"
            | "requires"
            | "return"
            | "short"
            | "signed"
            | "sizeof"
            | "static"
            | "static_assert"
            | "static_cast"
            | "struct"
            | "switch"
            | "template"
            | "this"
            | "thread_local"
            | "throw"
            | "true"
            | "try"
            | "typedef"
            | "typeid"
            | "typename"
            | "union"
            | "unsigned"
            | "using"
            | "virtual"
            | "void"
            | "volatile"
            | "wchar_t"
            | "while"
            | "xor"
            | "xor_eq"
    )
}
