<#
.SYNOPSIS
    Check Insta360 ONE RS USB detection status.
.DESCRIPTION
    Scans PnP devices, USB hubs, and disk drives for the Insta360 camera
    (VID_4255 / PID_0001) and reports connection status, errors, and
    troubleshooting suggestions.
#>

Write-Host "`n=== Insta360 ONE RS USB Detection Check ===" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

# --- 1. Check for Insta360 PnP device by Vendor ID ---
Write-Host "[1] Searching PnP devices for VID_4255 (Insta360)..." -ForegroundColor Yellow
$insta360Devices = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match 'VID_4255' }

if ($insta360Devices) {
    Write-Host "  FOUND Insta360 device(s):" -ForegroundColor Green
    $insta360Devices | ForEach-Object {
        Write-Host "    Name   : $($_.FriendlyName)" -ForegroundColor White
        Write-Host "    Status : $($_.Status)" -ForegroundColor $(if ($_.Status -eq 'OK') { 'Green' } else { 'Red' })
        Write-Host "    Class  : $($_.Class)"
        Write-Host "    ID     : $($_.InstanceId)"
        Write-Host ""
    }
} else {
    Write-Host "  NOT FOUND — no PnP device with VID_4255 detected." -ForegroundColor Red
}

# --- 2. Check for any USB mass storage / removable drives ---
Write-Host "[2] Checking for removable/USB disk drives..." -ForegroundColor Yellow
$usbDisks = Get-CimInstance -ClassName Win32_DiskDrive | Where-Object { $_.InterfaceType -eq 'USB' }

if ($usbDisks) {
    Write-Host "  USB disk(s) found:" -ForegroundColor Green
    $usbDisks | ForEach-Object {
        Write-Host "    Model : $($_.Model)"
        Write-Host "    Size  : $([math]::Round($_.Size / 1GB, 1)) GB"
        Write-Host "    Status: $($_.Status)"
        Write-Host ""
    }
} else {
    Write-Host "  No USB disk drives detected." -ForegroundColor DarkGray
}

# --- 3. Check for mounted volumes from USB ---
Write-Host "[3] Checking for drive letters from USB storage..." -ForegroundColor Yellow
$usbVolumes = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DriveType -eq 2 }

if ($usbVolumes) {
    Write-Host "  Removable drive(s):" -ForegroundColor Green
    $usbVolumes | ForEach-Object {
        $label = if ($_.VolumeName) { $_.VolumeName } else { "(no label)" }
        $freeGB = [math]::Round($_.FreeSpace / 1GB, 1)
        $totalGB = [math]::Round($_.Size / 1GB, 1)
        Write-Host "    $($_.DeviceID)\ — $label — $freeGB / $totalGB GB free"
    }
    Write-Host ""
} else {
    Write-Host "  No removable drives mounted." -ForegroundColor DarkGray
}

# --- 4. Check for .insv files on any removable drive ---
if ($usbVolumes) {
    Write-Host "[4] Scanning removable drives for .insv / .insp files..." -ForegroundColor Yellow
    foreach ($vol in $usbVolumes) {
        $dcim = Join-Path $vol.DeviceID "DCIM"
        if (Test-Path $dcim) {
            $instaFiles = Get-ChildItem -Path $dcim -Recurse -Include *.insv, *.insp -ErrorAction SilentlyContinue
            if ($instaFiles) {
                Write-Host "  Found $($instaFiles.Count) Insta360 file(s) on $($vol.DeviceID)\" -ForegroundColor Green
                $instaFiles | Select-Object -First 5 | ForEach-Object {
                    $sizeMB = [math]::Round($_.Length / 1MB, 1)
                    Write-Host "    $($_.Name)  ($sizeMB MB)  $($_.LastWriteTime)"
                }
                if ($instaFiles.Count -gt 5) {
                    Write-Host "    ... and $($instaFiles.Count - 5) more"
                }
            } else {
                Write-Host "  DCIM folder exists on $($vol.DeviceID)\ but no .insv/.insp files found." -ForegroundColor DarkGray
            }
        }
    }
    Write-Host ""
}

# --- 5. Check USB hub / controller error events ---
Write-Host "[5] Checking recent USB error events (last 30 min)..." -ForegroundColor Yellow
$since = (Get-Date).AddMinutes(-30)
$usbErrors = Get-WinEvent -FilterHashtable @{
    LogName   = 'System'
    Level     = 2, 3          # Error, Warning
    StartTime = $since
} -ErrorAction SilentlyContinue | Where-Object { $_.Message -match 'USB|port reset|hub' }

if ($usbErrors) {
    Write-Host "  Found $($usbErrors.Count) USB-related warning/error(s):" -ForegroundColor Red
    $usbErrors | Select-Object -First 5 | ForEach-Object {
        Write-Host "    [$($_.TimeCreated.ToString('HH:mm:ss'))] $($_.Message.Substring(0, [Math]::Min(120, $_.Message.Length)))" -ForegroundColor DarkYellow
    }
    Write-Host ""
} else {
    Write-Host "  No recent USB errors. Good." -ForegroundColor Green
}

# --- 6. List all USB controllers ---
Write-Host "[6] USB Host Controllers:" -ForegroundColor Yellow
$controllers = Get-PnpDevice -PresentOnly -Class USB | Where-Object { $_.FriendlyName -match 'Host Controller|Root Hub' }
$controllers | ForEach-Object {
    $statusColor = if ($_.Status -eq 'OK') { 'Green' } else { 'Red' }
    Write-Host "    [$($_.Status)] $($_.FriendlyName)" -ForegroundColor $statusColor
}

# --- Summary ---
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
if ($insta360Devices) {
    $okCount = ($insta360Devices | Where-Object { $_.Status -eq 'OK' }).Count
    if ($okCount -gt 0) {
        Write-Host "Insta360 detected and working (Status=OK)." -ForegroundColor Green
        Write-Host "Look for a removable drive letter above to access .insv files."
    } else {
        Write-Host "Insta360 detected but has errors — try a different USB port or reboot." -ForegroundColor Yellow
    }
} else {
    Write-Host "Insta360 NOT detected. Troubleshooting steps:" -ForegroundColor Red
    Write-Host "  1. Ensure the camera is powered ON (LCD may sleep — that's normal)"
    Write-Host "  2. Try a different USB-C cable (data cable, not charge-only)"
    Write-Host "  3. Try a different USB port (preferably USB 3.0 / blue port)"
    Write-Host "  4. Reboot the PC (clears USB controller state)"
    Write-Host "  5. On the camera: Settings > USB Mode > ensure 'USB Storage' is selected"
    Write-Host "  6. If still failing: transfer via phone (Insta360 app > Wi-Fi > export MP4)"
    Write-Host "  7. Last resort: USB SD card reader (~5 EUR)"
}
Write-Host ""
