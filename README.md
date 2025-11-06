Op 6 november 2025 werd het besluit op een verzoek om informatie over de voorbereiding van openbaar worden en digitaal beschikbaar stellen van het Centraal Archief Bijzondere Rechtspleging (CABR) gepubliceerd via https://www.rijksoverheid.nl/documenten/woo-besluiten/2025/11/06/besluit-op-woo-verzoek-over-voorbereiding-openbaar-worden-en-digitaal-beschikbaar-stellen-cabr

Naast het besluit en documentenlijst bevat de publicatie een tweetal ZIP bestanden van resp. 808 MB en 838 MB, in totaal 525 PDF bestanden.

Met het Python-script [pdf_qa_local.py](https://github.com/coret/cabr_woo_bevragen/blob/main/pdf_qa_local.py) worden bestanden in de pdfs directory geïndexeerd in een vector database (chromadb) en kun je deze via de commandline bevragen.

Het Python-script [pdf_qa_app.py](https://github.com/coret/cabr_woo_bevragen/blob/main/pdf_qa_app.py) wordt er op basis van Streamlit een chat omgeving gemaakt die lokaal draait en in je browser is te openen via `http://localhost:8501/`.

![screenshot chat](screenshot2.png "screenshot chat met context")
