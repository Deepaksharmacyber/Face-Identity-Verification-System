from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

OUTPUT_FILE = "docs/identity_specification.pdf"

styles = getSampleStyleSheet()

content = []

content.append(Paragraph("Identity Specification Document", styles['Title']))
content.append(Spacer(1,20))

content.append(Paragraph("1. Canonical Identity Definition", styles['Heading2']))
content.append(Paragraph(
"The canonical identity is defined using the master reference image "
"'master_identity.png'. This image represents the stable visual definition "
"of the identity used for all future comparisons.", styles['BodyText']))
content.append(Spacer(1,10))

content.append(Paragraph("2. Embedding Model", styles['Heading2']))
content.append(Paragraph(
"Face embeddings are generated using the InsightFace Buffalo_L model "
"(ArcFace architecture). The model extracts numerical feature vectors "
"representing facial identity.", styles['BodyText']))
content.append(Spacer(1,10))

content.append(Paragraph("3. Embedding Dimension", styles['Heading2']))
content.append(Paragraph(
"The embedding vector has a dimension of 512 and is normalized using "
"L2 normalization to ensure stable cosine similarity comparisons.", styles['BodyText']))
content.append(Spacer(1,10))

content.append(Paragraph("4. Validation Rules", styles['Heading2']))
content.append(Paragraph(
"Identity validation is performed using cosine similarity between the "
"reference embedding and the candidate image embedding.", styles['BodyText']))
content.append(Spacer(1,10))

content.append(Paragraph("5. Similarity Threshold", styles['Heading2']))
content.append(Paragraph(
"The system uses a similarity threshold of 0.93. Images producing a cosine "
"similarity score greater than or equal to this threshold are classified as "
"the same identity.", styles['BodyText']))
content.append(Spacer(1,10))

content.append(Paragraph("6. Approved Variation Space", styles['Heading2']))
content.append(Paragraph(
"The system allows controlled variations including small rotations, minor "
"lighting changes, and small contrast adjustments while preserving identity.", styles['BodyText']))
content.append(Spacer(1,10))

content.append(Paragraph("7. Drift Control", styles['Heading2']))
content.append(Paragraph(
"Sixteen controlled derivations were generated and validated against the "
"identity anchor. All derivations passed the similarity threshold, "
"demonstrating identity stability.", styles['BodyText']))

doc = SimpleDocTemplate(OUTPUT_FILE)
doc.build(content)

print("Identity specification PDF created:", OUTPUT_FILE)