$docx = "D:\Raspberry_Pi_Rust\docs\ANPR_Raspberry_Pi5_Optimization_Report.docx"
$pdf  = "D:\Raspberry_Pi_Rust\docs\ANPR_Raspberry_Pi5_Optimization_Report.pdf"

Write-Host "Opening Word COM to convert: $docx"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docx)
    # wdExportFormatPDF = 17
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close([ref]$false)
    Write-Host "PDF Export Succeeded!"
} catch {
    Write-Host "Error during export: $_"
} finally {
    $word.Quit()
}

if (Test-Path $pdf) {
    $item = Get-Item $pdf
    Write-Host "PDF created: $($item.FullName) ($($item.Length) bytes)"
} else {
    Write-Host "PDF file not found."
}
