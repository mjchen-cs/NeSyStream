# Netflix Open Content: two steps, because the masters are large and come in many formats.
#  1) list:     powershell -ExecutionPolicy Bypass -File download_netflix.ps1 -List
#               -> data\sources\netflix\netflix_listing.csv (all video files with size)
#  2) download: put the keys you want (one per line, copied from the csv) into data\sources\netflix\netflix_pick.txt
#               powershell -ExecutionPolicy Bypass -File download_netflix.ps1
# Prefer SDR 1080p/UHD .mp4/.mov files of a few GB; skip EXR/TIFF frame sequences and HDR/IMF packages.
param([switch]$List, [string]$Bucket = "download.opencontent.netflix.com")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the blue progress bar makes Invoke-WebRequest very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\sources\netflix"; New-Item -ItemType Directory -Force $out | Out-Null
# The bucket name contains dots, so https://<bucket>.s3.amazonaws.com fails certificate checks.
# Try path-style HTTPS first, then plain HTTP on the virtual host (public, read-only data).
$base = $null
foreach ($cand in @("https://s3.amazonaws.com/$Bucket", "https://s3.us-west-2.amazonaws.com/$Bucket", "https://s3.us-east-1.amazonaws.com/$Bucket", "http://$Bucket.s3.amazonaws.com")) {
  try { [xml]$probe = (Invoke-WebRequest -UseBasicParsing -Uri "$cand/?list-type=2&max-keys=1").Content
        if ($probe.ListBucketResult) { $base = $cand; break } } catch { }
}
if (-not $base) { Write-Host "[failed] no endpoint of bucket $Bucket answered."; exit 1 }
Write-Host "using $base"
if ($List) {
  $rows = @(); $tok = $null; $page = 0
  do {
    $u = "$base/?list-type=2&max-keys=1000"; if ($tok) { $u += "&continuation-token=" + [uri]::EscapeDataString($tok) }
    try { [xml]$x = (Invoke-WebRequest -UseBasicParsing -Uri $u).Content } catch { Write-Host "[failed] cannot list $base : $_"; Write-Host ""; exit 1 }
    foreach ($c in $x.ListBucketResult.Contents) {
      if ($c.Key -match "\.(mp4|mov|mxf|mkv|y4m)$") { $rows += [pscustomobject]@{key = $c.Key; GB = [math]::Round([double]$c.Size / 1GB, 2)} }
    }
    $tok = $x.ListBucketResult.NextContinuationToken; $page += 1
    Write-Host ("  page {0}: {1} video files so far" -f $page, $rows.Count)
  } while ($tok)
  $rows | Sort-Object key | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $out "netflix_listing.csv")
  Write-Host "listed $($rows.Count) video files -> $out\netflix_listing.csv  (edit netflix_pick.txt to choose files)"
  exit 0
}
$pickFile = Join-Path $out "netflix_pick.txt"
if (-not (Test-Path $pickFile)) { Write-Host "Run with -List first, then write the keys to download into $pickFile"; exit 1 }
foreach ($k in Get-Content $pickFile | Where-Object { $_.Trim() -ne "" }) {
  $dst = Join-Path $out ($k.Trim().Split("/")[-1])
  Write-Host "=== $k"
  & curl.exe -f -L -C - --retry 5 -o $dst ("$base/" + ($k.Trim() -replace " ", "%20"))
  if ($LASTEXITCODE -ne 0) { Write-Host "[failed] $k" }
}
Write-Host "Done."
