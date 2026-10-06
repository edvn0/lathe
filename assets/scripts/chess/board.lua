-- Board geometry. Squares are integers 0..63 with a1 = 0 (file = sq % 8, rank = sq // 8); the board is centred on the
-- origin with white on the -z side.

local board = {}

board.square_size = 1.08
board.top_y = 0.3

function board.file(square)
    return square % 8
end

function board.rank(square)
    return square // 8
end

function board.square(file, rank)
    return rank * 8 + file
end

-- The centre of the square's top face, as x, y, z.
function board.position(square)
    return (board.file(square) - 3.5) * board.square_size,
        board.top_y,
        (board.rank(square) - 3.5) * board.square_size
end

function board.name(square)
    return string.char(97 + board.file(square)) .. tostring(board.rank(square) + 1)
end

-- The square under a point on the board plane, or nil off the board.
function board.from_world(x, z)
    local file = math.floor(x / board.square_size + 4.0)
    local rank = math.floor(z / board.square_size + 4.0)

    if file < 0 or file > 7 or rank < 0 or rank > 7 then
        return nil
    end

    return board.square(file, rank)
end

return board
