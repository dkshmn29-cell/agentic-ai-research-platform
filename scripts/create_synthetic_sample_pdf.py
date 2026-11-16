import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sample_banking_genai_research.pdf"

PAGES = [
    (
        "Overview of a fictional pilot",
        [
            "This document is entirely synthetic. The fictional organization, scenario, processes, and statements below were invented for testing PDF extraction, chunking, and citation handling. Nothing here is factual market research.",
            "Northstar Community Bank is fictional. In this scenario, a team tests an internal text assistant that searches approved help articles and suggests draft responses for customer-service employees. A human employee reviews and sends every response. The assistant cannot approve payments or change customer records.",
            "The fictional pilot tests three tasks: finding a procedure, summarizing an article, and drafting a plain-language explanation. This document reports no measured performance results.",
        ],
    ),
    (
        "Fictional operations workflow",
        [
            "An operations team uses a retrieval assistant to find sections in procedure manuals. A separate summarization feature condenses internal memos. Before relying on an answer, an employee opens the cited source and verifies its version and date.",
            "The fictional team has not measured productivity, accuracy, cost, or adoption. The examples describe imagined workflows only and are not claims about banks or the banking industry.",
            "Data note: Every organization and example in this file is invented. Do not cite this sample as factual evidence.",
        ],
    ),
    (
        "Fictional risk and review notes",
        [
            "The fictional project team tracks incorrect summaries, outdated source material, exposure of confidential information, and retrieved passages that contain instructions attempting to change the assistant's behavior.",
            "Proposed controls in this fictional scenario include restricting retrieval to approved documents, displaying source filenames and page numbers, requiring employee approval, avoiding unnecessary customer-data logging, and testing questions with no supporting evidence.",
            "Limitation: This synthetic fixture cannot support real claims about banking, generative AI performance, adoption, regulations, or risk rates.",
        ],
    ),
]


def create_sample_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = letter
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    pdf.setTitle("Synthetic Banking GenAI Research Test Fixture")
    pdf.setAuthor("Agentic AI Research Platform test data")

    for page_number, (title, paragraphs) in enumerate(PAGES, start=1):
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(54, page_height - 48, "SYNTHETIC TEST DATA - NOT REAL RESEARCH")
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(54, page_height - 82, title)

        current_y = page_height - 122
        pdf.setFont("Helvetica", 10)
        for paragraph in paragraphs:
            for line in textwrap.wrap(paragraph, width=94):
                pdf.drawString(54, current_y, line)
                current_y -= 14
            current_y -= 12

        pdf.setFont("Helvetica", 9)
        pdf.drawString(54, 35, f"Fictional software fixture | Page {page_number} of {len(PAGES)}")
        pdf.showPage()

    pdf.save()
    return output_path


if __name__ == "__main__":
    print(f"Created synthetic PDF: {create_sample_pdf()}")