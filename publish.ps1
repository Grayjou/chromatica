# PowerShell script to clean, build, and upload chromatica package
Write-Host "=== Cleaning previous build artifacts ==="
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

Write-Host "=== Checking version number ==="
$pyproject = Get-Content -Path "pyproject.toml" | Out-String
if ($pyproject -match 'version\s*=\s*"(.*?)"') {
    $version = $Matches[1]
    Write-Host "Chromatica version detected: $version"
} else {
    Write-Host "Could not find version in pyproject.toml"
    exit 1
}

Write-Host "=== Building the package ==="
python -m build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. Aborting."
    exit 1
}

Write-Host "=== Uploading to PyPI ==="
python -m twine upload dist/*

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully uploaded Chromatica v$version to PyPI!"
} else {
    Write-Host "❌ Upload failed. Check errors above."
    exit 1
}