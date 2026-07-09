param(
  [string]$Root = "D:\tunlipsyn_runpod_github_upload",
  [string]$Owner = "luongvantuankb2-collab",
  [string]$Repo = "tunlipsyn-runpod-serverless",
  [string]$Branch = "main",
  [string]$Message = "Update RunPod serverless files"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $Root)) {
  throw "Root folder not found: $Root"
}

$token = $env:GITHUB_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) {
  $secure = Read-Host "Paste GitHub token (repo permission)" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
  finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
if ([string]::IsNullOrWhiteSpace($token)) { throw "Missing GitHub token" }

$headers = @{
  Authorization = "Bearer $token"
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
  "User-Agent" = "tunlipsyn-uploader"
}

function Invoke-GitHubJson {
  param([string]$Method, [string]$Uri, $Body = $null)
  if ($null -eq $Body) {
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $headers
  }
  $json = $Body | ConvertTo-Json -Depth 20
  return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $headers -ContentType "application/json; charset=utf-8" -Body $json
}

$files = Get-ChildItem -LiteralPath $Root -Recurse -File -Force | Where-Object {
  $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc'
}

Write-Host "Uploading $($files.Count) files to $Owner/$Repo branch $Branch"

foreach ($file in $files) {
  $relative = $file.FullName.Substring((Resolve-Path -LiteralPath $Root).Path.Length).TrimStart('\') -replace '\\','/'
  $encodedPath = ($relative -split '/' | ForEach-Object { [uri]::EscapeDataString($_) }) -join '/'
  $uri = "https://api.github.com/repos/$Owner/$Repo/contents/$encodedPath"
  $sha = $null
  try {
    $existing = Invoke-GitHubJson -Method GET -Uri "$uri`?ref=$Branch"
    $sha = $existing.sha
  } catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 404) { throw }
  }

  $content = [Convert]::ToBase64String([IO.File]::ReadAllBytes($file.FullName))
  $body = @{
    message = "$Message`: $relative"
    content = $content
    branch = $Branch
  }
  if ($sha) { $body.sha = $sha }

  Write-Host " -> $relative"
  Invoke-GitHubJson -Method PUT -Uri $uri -Body $body | Out-Null
}

Write-Host "DONE. Now open Actions and re-run the build."
