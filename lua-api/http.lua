---@meta

---Asynchronous HTTP for scripts. Requests run on worker threads and are finished on the main thread: poll a request
---with `done()`, or give `on_done`, which is called from the engine's update with the finished request. https only, and
---only to the hosts the engine allows (raw.githubusercontent.com, github.com and objects.githubusercontent.com).
---@class http
http = {}

---@class HttpOptions
---@field sha256? string 64 hex characters; a body whose hash differs is an error
---@field on_done? fun(request: HttpRequest) called once the request has finished (not after cancel())

---@class HttpDownloadOptions
---@field sha256 string 64 hex characters; required. A cached file that already matches is not downloaded again
---@field name? string where under the download cache to keep it (letters, digits, . _ - and / between parts); default is the hash plus the url's extension
---@field on_done? fun(request: HttpRequest)

---An in-flight or finished request.
---@class HttpRequest
local HttpRequest = {}

---Fetches a url into memory (at most 4 MiB). Raises an error for a url that is not https, a host that is not allowed
---or too many requests in flight (8).
---@param url string
---@param options? HttpOptions
---@return HttpRequest
function http.get(url, options) end

---Downloads a url into the cache, checks its sha256 and keeps it. Pass the request to assets.load_model once it is ok.
---@param url string
---@param options HttpDownloadOptions
---@return HttpRequest
function http.download(url, options) end

---How many requests are in flight.
---@return integer
function http.pending() end

---True once the request has finished, whether it worked or not.
---@return boolean
function HttpRequest:done() end
---True when the request finished and succeeded.
---@return boolean
function HttpRequest:ok() end
---Why a finished request failed ("hash_mismatch", "timeout", "unsuccessful_status_code (status 404)", "canceled", ...), or nil.
---@return string?
function HttpRequest:error() end
---The HTTP status of a finished request (0 when none was received).
---@return integer
function HttpRequest:status() end
---The body of a finished http.get that succeeded, or nil.
---@return string?
function HttpRequest:body() end
---Where a finished http.download that succeeded saved its file, or nil.
---@return string?
function HttpRequest:path() end
---Stops a request that has not finished. Its on_done is not called.
function HttpRequest:cancel() end
