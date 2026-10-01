use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::{
    env,
    ffi::{OsStr, OsString},
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    sync::OnceLock,
};

const IMAGE: &str = "cross-build:latest";
const BUILD_JOBS: usize = 20;

const ENVIRONMENT_HELP: &str = r#"
Environment variables:

  CMAKE_BUILD_TYPE
      Debug
      Release
      RelWithDebInfo
      MinSizeRel

      Default: Debug

  LINKER
      default
      mold

      Default: default

  RENDERDOC_INCLUDE_PATH
      Host directory containing renderdoc_app.h.

  PROFILE_BUILD=1
      Adds -fno-omit-frame-pointer.

  SANITIZE=1
      Enables LATHE_SANITIZE.

  WERROR=1
      Enables LATHE_WERROR.

  PROFILER
      perf
      perf-record
      callgrind

      Default: perf

  EXECUTABLE_NAME
      Binary name relative to <build>/bin.

      Default: lathe

Examples:

  cargo xtask rebuild

  CMAKE_BUILD_TYPE=RelWithDebInfo \
    LINKER=mold \
    cargo xtask rebuild

  CMAKE_BUILD_TYPE=Debug \
    SANITIZE=1 \
    cargo xtask rebuild

  cargo xtask tidy -- -fix

  CMAKE_BUILD_TYPE=RelWithDebInfo \
    PROFILE_BUILD=1 \
    PROFILER=perf-record \
    cargo xtask rebuild

  CMAKE_BUILD_TYPE=RelWithDebInfo \
    PROFILE_BUILD=1 \
    PROFILER=perf-record \
    cargo xtask profile
"#;

#[derive(Parser)]
#[command(
    name = "xtask",
    about = "Lathe build and development tasks",
    after_help = ENVIRONMENT_HELP,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Configure the build using CMake.
    Configure,

    /// Build the existing configuration.
    Build,

    /// Remove the build directory, configure, and build.
    Rebuild,

    /// Remove the selected build directory.
    Clean,

    /// Open an interactive shell inside the build container.
    Shell,

    /// Build lathe-tests and run CTest.
    Test {
        /// Extra arguments passed to CTest.
        #[arg(last = true)]
        args: Vec<OsString>,
    },

    /// Build and run clang-tidy.
    Tidy {
        /// Extra arguments passed to run_clang_tidy.sh.
        #[arg(last = true)]
        args: Vec<OsString>,
    },

    /// Run the executable under the selected profiler on the host.
    Profile {
        /// Arguments passed to the executable.
        #[arg(last = true)]
        args: Vec<OsString>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Linker {
    Default,
    Mold,
}

impl Linker {
    fn from_env() -> Result<Self> {
        match env::var("LINKER")
            .unwrap_or_else(|_| "default".to_owned())
            .as_str()
        {
            "default" => Ok(Self::Default),
            "mold" => Ok(Self::Mold),
            value => bail!("unknown LINKER: {value} (expected default or mold)"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Mold => "mold",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Profiler {
    Perf,
    PerfRecord,
    Callgrind,
}

impl Profiler {
    fn from_env() -> Result<Self> {
        match env::var("PROFILER")
            .unwrap_or_else(|_| "perf".to_owned())
            .as_str()
        {
            "perf" => Ok(Self::Perf),
            "perf-record" => Ok(Self::PerfRecord),
            "callgrind" => Ok(Self::Callgrind),
            value => bail!(
                "unknown PROFILER: {value} \
                 (expected perf, perf-record, or callgrind)"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Perf => "perf",
            Self::PerfRecord => "perf-record",
            Self::Callgrind => "callgrind",
        }
    }
}

struct Config {
    project_dir: PathBuf,
    build_type: String,
    renderdoc_include_path: Option<PathBuf>,
    cpm_cache_dir: PathBuf,
    container_home: PathBuf,
    linker: Linker,
    profiler: Profiler,
    executable_name: String,
    build_dir: PathBuf,
    profile_build: bool,
    sanitize: bool,
    werror: bool,
}

impl Config {
    fn load() -> Result<Self> {
        let project_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .context("failed to locate repository root")?;

        let home = env::var_os("HOME")
            .map(PathBuf::from)
            .context("HOME is not set")?;

        let build_type = env::var("CMAKE_BUILD_TYPE").unwrap_or_else(|_| "Debug".to_owned());

        validate_build_type(&build_type)?;

        let build_dir = PathBuf::from(format!("build/{}", build_type.to_ascii_lowercase()));

        let renderdoc_include_path = env::var_os("RENDERDOC_INCLUDE_PATH")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);

        Ok(Self {
            project_dir,
            build_type,
            renderdoc_include_path,
            cpm_cache_dir: home.join(".cache/CPM"),
            container_home: home.join(".cache/cross-build-container-home"),
            linker: Linker::from_env()?,
            profiler: Profiler::from_env()?,
            executable_name: env::var("EXECUTABLE_NAME").unwrap_or_else(|_| "lathe".to_owned()),
            build_dir,
            profile_build: env_flag("PROFILE_BUILD"),
            sanitize: env_flag("SANITIZE"),
            werror: env_flag("WERROR"),
        })
    }

    fn print_summary(&self) {
        println!("Build configuration:");
        println!("  type:       {}", self.build_type);
        println!("  linker:     {}", self.linker.name());
        println!("  build dir:  {}", self.build_dir.display());

        if self.profile_build {
            println!("  profile:    enabled");
        }

        if self.sanitize {
            println!("  sanitize:   enabled");
        }

        if self.werror {
            println!("  werror:     enabled");
        }

        if let Some(renderdoc_include_path) = &self.renderdoc_include_path {
            println!("  renderdoc:  {}", renderdoc_include_path.display());
        }
    }

    fn validate_renderdoc_path(&self) -> Result<()> {
        let Some(path) = &self.renderdoc_include_path else {
            return Ok(());
        };

        if !path.is_dir() {
            bail!(
                "RENDERDOC_INCLUDE_PATH is not a directory:\n  {}",
                path.display()
            );
        }

        let header = path.join("renderdoc_app.h");

        if !header.is_file() {
            bail!("renderdoc_app.h was not found:\n  {}", header.display());
        }

        Ok(())
    }

    fn prepare_container(&self) -> Result<()> {
        fs::create_dir_all(&self.cpm_cache_dir).context("failed to create CPM cache directory")?;

        fs::create_dir_all(&self.container_home)
            .context("failed to create container HOME directory")?;

        self.validate_renderdoc_path()
    }

    fn container_command(&self, interactive: bool) -> Result<Command> {
        self.prepare_container()?;

        let mut command = Command::new("docker");

        command.arg("run").arg("--rm").arg("--init");

        if interactive {
            command.arg("-it");
        }

        command
            .arg("--mount")
            .arg(bind_mount(&self.project_dir, &self.project_dir, false))
            .arg("--mount")
            .arg(bind_mount(&self.cpm_cache_dir, &self.cpm_cache_dir, false))
            .arg("--mount")
            .arg(bind_mount(
                &self.container_home,
                &self.container_home,
                false,
            ))
            .arg("--env")
            .arg(format!("CPM_SOURCE_CACHE={}", self.cpm_cache_dir.display()))
            .arg("--env")
            .arg(format!("HOME={}", self.container_home.display()))
            .arg("--workdir")
            .arg(&self.project_dir);

        if !is_rootless_docker() {
            let uid = command_output("id", ["-u"])?;
            let gid = command_output("id", ["-g"])?;

            command.arg("--user").arg(format!("{uid}:{gid}"));
        }

        if let Some(renderdoc_include_path) = &self.renderdoc_include_path {
            command.arg("--mount").arg(bind_mount(
                renderdoc_include_path,
                renderdoc_include_path,
                true,
            ));
        }

        command.arg(IMAGE);

        Ok(command)
    }

    fn run_container<I, S>(&self, args: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut command = self.container_command(false)?;
        command.args(args);

        run_checked(&mut command)
    }

    fn container_status<I, S>(&self, args: I) -> Result<ExitStatus>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut command = self.container_command(false)?;
        command.args(args);

        command.status().context("failed to start Docker container")
    }

    fn configure(&self) -> Result<()> {
        self.print_summary();

        let mut args = vec![
            OsString::from("cmake"),
            OsString::from("-S"),
            self.project_dir.as_os_str().to_owned(),
            OsString::from("-B"),
            self.project_dir.join(&self.build_dir).into_os_string(),
            OsString::from("-G"),
            OsString::from("Ninja"),
            OsString::from(format!("-DCMAKE_BUILD_TYPE={}", self.build_type)),
            OsString::from("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"),
        ];

        if self.linker == Linker::Mold {
            let status = self.container_status(["sh", "-c", "command -v mold >/dev/null 2>&1"])?;

            if !status.success() {
                bail!("LINKER=mold requested, but mold is not installed in {IMAGE}");
            }

            args.push(OsString::from("-DCMAKE_LINKER_TYPE=MOLD"));
        }

        if let Some(renderdoc_include_path) = &self.renderdoc_include_path {
            args.push(OsString::from(format!(
                "-DRENDERDOC_INCLUDE_PATH={}",
                renderdoc_include_path.display()
            )));
        }

        if self.profile_build {
            args.push(OsString::from("-DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer"));
            args.push(OsString::from("-DCMAKE_C_FLAGS=-fno-omit-frame-pointer"));
        }

        if self.sanitize {
            args.push(OsString::from("-DLATHE_SANITIZE=ON"));
        }

        if self.werror {
            args.push(OsString::from("-DLATHE_WERROR=ON"));
        }

        self.run_container(args)?;
        self.update_compile_commands_link()
    }

    fn build(&self) -> Result<()> {
        self.run_container([
            OsString::from("cmake"),
            OsString::from("--build"),
            self.project_dir.join(&self.build_dir).into_os_string(),
            OsString::from(format!("-j{BUILD_JOBS}")),
        ])?;

        self.update_compile_commands_link()
    }

    fn rebuild(&self) -> Result<()> {
        self.clean()?;
        self.configure()?;
        self.build()
    }

    fn clean(&self) -> Result<()> {
        let build_path = self.project_dir.join(&self.build_dir);

        if build_path.exists() {
            fs::remove_dir_all(&build_path)
                .with_context(|| format!("failed to remove {}", build_path.display()))?;
        }

        let compile_commands = self.project_dir.join("compile_commands.json");

        if let Ok(current_target) = fs::read_link(&compile_commands) {
            let expected_target = self.build_dir.join("compile_commands.json");

            if current_target == expected_target {
                fs::remove_file(&compile_commands)
                    .with_context(|| format!("failed to remove {}", compile_commands.display()))?;
            }
        }

        Ok(())
    }

    fn shell(&self) -> Result<()> {
        let mut command = self.container_command(true)?;
        command.arg("bash");

        run_checked(&mut command)
    }

    fn test(&self, extra_args: &[OsString]) -> Result<()> {
        self.run_container([
            OsString::from("cmake"),
            OsString::from("--build"),
            self.project_dir.join(&self.build_dir).into_os_string(),
            OsString::from("--target"),
            OsString::from("lathe-tests"),
            OsString::from(format!("-j{BUILD_JOBS}")),
        ])?;

        let mut args = vec![
            OsString::from("ctest"),
            OsString::from("--parallel"),
            OsString::from(BUILD_JOBS.to_string()),
            OsString::from("--test-dir"),
            self.project_dir.join(&self.build_dir).into_os_string(),
            OsString::from("--output-on-failure"),
        ];

        args.extend_from_slice(extra_args);

        self.run_container(args)
    }

    fn tidy(&self, extra_args: &[OsString]) -> Result<()> {
        // Some sources include headers generated during the build.
        self.build()?;

        let mut args = vec![
            self.project_dir
                .join("tools/run_clang_tidy.sh")
                .into_os_string(),
            self.project_dir.join(&self.build_dir).into_os_string(),
        ];

        args.extend_from_slice(extra_args);

        self.run_container(args)
    }

    fn profile(&self, executable_args: &[OsString]) -> Result<()> {
        let executable = self
            .project_dir
            .join(&self.build_dir)
            .join("bin")
            .join(&self.executable_name);

        if !is_executable(&executable) {
            bail!(
                "executable not found: {}\n\
                 Build it first with:\n\
                 CMAKE_BUILD_TYPE={} cargo xtask build",
                executable.display(),
                self.build_type
            );
        }

        println!("Profiler: {}", self.profiler.name());

        match self.profiler {
            Profiler::Perf => {
                let mut command = Command::new("perf");

                command
                    .arg("stat")
                    .arg("-d")
                    .arg("--")
                    .arg(&executable)
                    .args(executable_args);

                run_checked(&mut command)
            }

            Profiler::PerfRecord => {
                let output = self.project_dir.join(&self.build_dir).join("perf.data");

                let graph_mode = if self.profile_build { "fp" } else { "dwarf" };

                let mut command = Command::new("perf");

                command
                    .arg("record")
                    .arg("-g")
                    .arg("--call-graph")
                    .arg(graph_mode)
                    .arg("-o")
                    .arg(&output)
                    .arg("--")
                    .arg(&executable)
                    .args(executable_args);

                run_checked(&mut command)?;

                println!("Report with:");
                println!("  perf report -i {}", output.display());

                Ok(())
            }

            Profiler::Callgrind => {
                let output = self
                    .project_dir
                    .join(&self.build_dir)
                    .join("callgrind.out.%p");

                let mut command = Command::new("valgrind");

                command
                    .arg("--tool=callgrind")
                    .arg(format!("--callgrind-out-file={}", output.display()))
                    .arg("--")
                    .arg(&executable)
                    .args(executable_args);

                run_checked(&mut command)?;

                println!("Open the resulting callgrind.out.<pid> file with kcachegrind");

                Ok(())
            }
        }
    }

    fn update_compile_commands_link(&self) -> Result<()> {
        let source = self
            .project_dir
            .join(&self.build_dir)
            .join("compile_commands.json");

        if !source.is_file() {
            return Ok(());
        }

        let destination = self.project_dir.join("compile_commands.json");

        if fs::symlink_metadata(&destination).is_ok() {
            fs::remove_file(&destination)
                .with_context(|| format!("failed to remove {}", destination.display()))?;
        }

        let relative_target = self.build_dir.join("compile_commands.json");

        symlink(&relative_target, &destination).with_context(|| {
            format!(
                "failed to create {} -> {}",
                destination.display(),
                relative_target.display()
            )
        })?;

        Ok(())
    }
}

fn validate_build_type(build_type: &str) -> Result<()> {
    match build_type {
        "Debug" | "Release" | "RelWithDebInfo" | "MinSizeRel" => Ok(()),
        value => bail!(
            "unknown CMAKE_BUILD_TYPE: {value} \
             (expected Debug, Release, RelWithDebInfo, or MinSizeRel)"
        ),
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name).as_deref() == Ok("1")
}

fn bind_mount(source: &Path, target: &Path, read_only: bool) -> String {
    let mut value = format!(
        "type=bind,source={},target={}",
        source.display(),
        target.display()
    );

    if read_only {
        value.push_str(",readonly");
    }

    value
}

fn is_rootless_docker() -> bool {
    static ROOTLESS: OnceLock<bool> = OnceLock::new();

    *ROOTLESS.get_or_init(|| {
        let Ok(output) = Command::new("docker")
            .args(["info", "--format", "{{.SecurityOptions}}"])
            .output()
        else {
            return false;
        };

        output.status.success() && String::from_utf8_lossy(&output.stdout).contains("name=rootless")
    })
}

fn command_output<const N: usize>(executable: &str, args: [&str; N]) -> Result<String> {
    let output = Command::new(executable)
        .args(args)
        .output()
        .with_context(|| format!("failed to execute {executable}"))?;

    if !output.status.success() {
        bail!("{executable} exited with {}", output.status);
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };

    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
}

fn run_checked(command: &mut Command) -> Result<()> {
    let description = format!("{command:?}");

    let status = command
        .status()
        .with_context(|| format!("failed to start {description}"))?;

    if !status.success() {
        bail!("{description} exited with {status}");
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;

    match cli.command {
        Task::Configure => config.configure(),
        Task::Build => config.build(),
        Task::Rebuild => config.rebuild(),
        Task::Clean => config.clean(),
        Task::Shell => config.shell(),
        Task::Test { args } => config.test(&args),
        Task::Tidy { args } => config.tidy(&args),
        Task::Profile { args } => config.profile(&args),
    }
}
