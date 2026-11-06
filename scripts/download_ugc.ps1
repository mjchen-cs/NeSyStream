# Download a category-balanced subset of YouTube-UGC (CC BY) 1080p originals into data\sources\ugc
# Uses the public Google Cloud Storage listing of the dataset bucket, so no file names are guessed.
#   powershell -ExecutionPolicy Bypass -File download_ugc.ps1 -List          # only write ugc_listing.csv
#   powershell -ExecutionPolicy Bypass -File download_ugc.ps1 -PerCategory 2 # download 2 per category
param([switch]$List, [int]$PerCategory = 2, [string]$Res = "1080P",
      [string]$Bucket = "ugc-dataset", [string]$Prefix = "original_videos/",
      [string]$Exclude = "HDR,VR,Lyrics")   # HDR needs tone mapping, VR is 360-degree, Lyrics duplicates LyricVideo
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the blue progress bar makes Invoke-WebRequest very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\sources\ugc"; New-Item -ItemType Directory -Force $out | Out-Null
$items = @(); $token = $null
do {
  $u = "https://storage.googleapis.com/storage/v1/b/$Bucket/o?prefix=$Prefix&maxResults=1000&fields=items(name,size),nextPageToken"
  if ($token) { $u += "&pageToken=$token" }
  try { $r = Invoke-RestMethod -Uri $u } catch { Write-Host "[failed] cannot list gs://$Bucket/$Prefix : $_"; Write-Host ""; exit 1 }
  $items += $r.items; $token = $r.nextPageToken
} while ($token)
$vid = $items | Where-Object { $_.name -match "/$Res/" -and $_.name -match "\.(mp4|mkv|webm|mov)$" }
$rows = $vid | ForEach-Object { $p = $_.name.Split("/"); [pscustomobject]@{category = $p[1]; name = $_.name; MB = [math]::Round([double]$_.size / 1MB, 1)} }
$ex = $Exclude.Split(",") | ForEach-Object { $_.Trim() }
$rows = $rows | Where-Object { $ex -notcontains $_.category }
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $out "ugc_listing.csv")
Write-Host ("found {0} files at {1} in {2} categories -> ugc_listing.csv" -f $rows.Count, $Res, ($rows.category | Sort-Object -Unique).Count)
if ($List -or $rows.Count -eq 0) { exit 0 }
$pick = $rows | Group-Object category | ForEach-Object { $_.Group | Sort-Object name | Select-Object -First $PerCategory }
foreach ($f in $pick) {
  $dst = Join-Path $out ($f.name.Split("/")[-1])
  Write-Host ("=== [{0}] {1} ({2} MB)" -f $f.category, $f.name, $f.MB)
  & curl.exe -f -L -C - --retry 5 -o $dst ("https://storage.googleapis.com/$Bucket/" + $f.name)
  if ($LASTEXITCODE -ne 0) { Write-Host "[failed] $($f.name)" }
}
Write-Host "Done: $($pick.Count) files in $out"
