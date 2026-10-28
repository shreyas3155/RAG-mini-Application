"""
Generates a multi-page sample PDF file for testing the RAG chatbot.
"""

def generate_pdf():
    # Basic PDF 1.4 template generator
    content_page1 = """BT
/F1 18 Tf
50 720 Td
(MERN Stack Notes - Part 1: Express.js & Middleware) Tj
/F1 12 Tf
0 -40 Td
(1. What is Middleware in Express.js?) Tj
0 -20 Td
(Middleware functions are functions that have access to the request object \(req\),) Tj
0 -15 Td
(the response object \(res\), and the next middleware function in the application's) Tj
0 -15 Td
(request-response cycle. These functions can perform tasks such as executing code,) Tj
0 -15 Td
(making changes to request and response objects, ending the request-response cycle,) Tj
0 -15 Td
(and calling the next middleware function using next\(\).) Tj
0 -30 Td
(2. Types of Middleware in Express:) Tj
0 -20 Td
(- Application-level middleware: Bound to an instance of express app using app.use\(\).) Tj
0 -15 Td
(- Router-level middleware: Bound to an instance of express.Router\(\).) Tj
0 -15 Td
(- Error-handling middleware: Defined with 4 arguments \(err, req, res, next\).) Tj
0 -15 Td
(- Built-in middleware: express.json\(\), express.urlencoded\(\), express.static\(\).) Tj
0 -15 Td
(- Third-party middleware: cors, morgan, helmet, cookie-parser.) Tj
ET"""

    content_page2 = """BT
/F1 18 Tf
50 720 Td
(MERN Stack Notes - Part 2: JWT Authentication) Tj
/F1 12 Tf
0 -40 Td
(1. What is JSON Web Token \(JWT\)?) Tj
0 -20 Td
(JWT is an open standard \(RFC 7519\) that defines a compact and self-contained way) Tj
0 -15 Td
(for securely transmitting information between parties as a JSON object.) Tj
0 -20 Td
(2. Anatomy of a JWT:) Tj
0 -15 Td
(- Header: Contains the token type \(JWT\) and signing algorithm \(e.g., HS256\).) Tj
0 -15 Td
(- Payload: Contains claims, user ID, role, and expiration time \(exp\).) Tj
0 -15 Td
(- Signature: Created by hashing encoded header + payload with a secret key.) Tj
0 -30 Td
(3. Access Token vs Refresh Token Strategy:) Tj
0 -20 Td
(Access tokens have a short lifespan \(e.g., 15 minutes\) to minimize attack risk.) Tj
0 -15 Td
(Refresh tokens have a longer lifespan \(e.g., 7 days\) and are stored securely) Tj
0 -15 Td
(in httpOnly cookies to reissue new access tokens when they expire.) Tj
ET"""

    content_page3 = """BT
/F1 18 Tf
50 720 Td
(MERN Stack Notes - Part 3: MongoDB vs PostgreSQL) Tj
/F1 12 Tf
0 -40 Td
(1. Key Differences:) Tj
0 -20 Td
(- Data Model: MongoDB is document-oriented \(JSON/BSON\); PostgreSQL is relational \(tables & rows\).) Tj
0 -15 Td
(- Schema: MongoDB offers flexible, dynamic schema; PostgreSQL enforces strict schemas and constraints.) Tj
0 -15 Td
(- Transactions: PostgreSQL provides mature ACID transactions out of the box; MongoDB added multi-doc ACID in v4.0.) Tj
0 -15 Td
(- Scaling: MongoDB scales horizontally via sharding; PostgreSQL typically scales vertically with read-replicas.) Tj
0 -30 Td
(2. Indexing in MongoDB:) Tj
0 -20 Td
(MongoDB uses B-Trees for indexes. Single field, compound indexes, and multikey indexes) Tj
0 -15 Td
(speed up queries significantly by avoiding full collection scans \(COLLSCAN\).) Tj
ET"""

    def make_page(content, page_num):
        content_bytes = content.encode("latin1")
        length = len(content_bytes)
        return f"""<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents {page_num * 2 + 3} 0 R
/Resources <<
  /Font <<
    /F1 <<
      /Type /Font
      /Subtype /Type1
      /BaseFont /Helvetica
    >>
  >>
>>
>>
endobj
{page_num * 2 + 3} 0 obj
<< /Length {length} >>
stream
{content}
endstream
endobj
"""

    p1 = make_page(content_page1, 1)
    p2 = make_page(content_page2, 2)
    p3 = make_page(content_page3, 3)

    pdf_body = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 5 0 R 7 0 R] /Count 3 >>
endobj
3 0 obj
{p1}
5 0 obj
{p2}
7 0 obj
{p3}
xref
0 10
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
trailer
<< /Size 10 /Root 1 0 R >>
startxref
120
%%EOF"""

    with open("sample_mern_notes.pdf", "wb") as f:
        f.write(pdf_body.encode("latin1"))

    print("Created sample_mern_notes.pdf successfully!")


if __name__ == "__main__":
    generate_pdf()
