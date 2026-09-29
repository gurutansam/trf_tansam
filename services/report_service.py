import io
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, PageBreak, TableStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from openpyxl import Workbook

def build_pdf(camera: str, location: str, sf: str, ef: str, summary, details, total: int, video_name: str | None = None) -> io.BytesIO:
    """Generate a formatted PDF report with summary metrics and detailed logs."""
    buffer = io.BytesIO()
    PW, PH = landscape(A4)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=(PW, PH),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    styles = getSampleStyleSheet()
    el = [Paragraph("<b>Traffic Detection Report</b>", styles["Title"])]
    if video_name:
        el.append(Paragraph(f"<b>Video:</b> {video_name}", styles["Normal"]))
    el += [
        Paragraph(f"<b>Camera:</b> {camera}", styles["Normal"]),
        Paragraph(f"<b>Location:</b> {location}", styles["Normal"]),
        Paragraph(f"<b>Duration:</b> {sf} → {ef}", styles["Normal"]),
        Paragraph(f"<b>Total Vehicle Count:</b> {total}", styles["Normal"]),
        Paragraph("<br/>", styles["Normal"]),
    ]
    st = Table([["Vehicle", "Count"]] + list(summary),
               colWidths=[PW * 0.6, PW * 0.2], hAlign="LEFT")
    st.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))
    el.append(st)
    el.append(PageBreak())
    dt = Table([["Time", "Camera", "Vehicle", "Location"]] + list(details),
               colWidths=[PW * 0.25, PW * 0.15, PW * 0.30, PW * 0.20], repeatRows=1, hAlign="LEFT")
    dt.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    el.append(dt)
    doc.build(el)
    buffer.seek(0)
    return buffer

def build_excel(camera: str, location: str, sf: str, ef: str, summary, details, total: int, video_name: str | None = None) -> io.BytesIO:
    """Generate an Excel spreadsheet with Summary and Details tabs."""
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Summary"
    if video_name:
        ws1.append(["Video", video_name])
    ws1.append(["Camera", camera])
    ws1.append(["Location", location])
    ws1.append(["Duration", f"{sf} → {ef}"])
    ws1.append(["Total Vehicle Count", total])
    ws1.append([])
    ws1.append(["Vehicle", "Count"])
    for row in summary:
        ws1.append(list(row))

    ws2 = wb.create_sheet(title="Details")
    if video_name:
        ws2.append(["Video", video_name])
    ws2.append(["Camera", camera])
    ws2.append(["Location", location])
    ws2.append(["Duration", f"{sf} → {ef}"])
    ws2.append(["Total Vehicle Count", total])
    ws2.append([])
    ws2.append(["Time", "Camera", "Vehicle", "Location"])
    for row in details:
        ws2.append(list(row))

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output
