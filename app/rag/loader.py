"""
loader.py
──────────
Loads raw files into LangChain `Document` objects.

Supported types: .csv, .pdf, .txt
Returns: list[Document]  — each Document has .page_content and .metadata

To add support for a new file type:
    1. Add the extension to SUPPORTED_EXTENSIONS below.
    2. Add a matching branch in load_document() with the loader implementation.
    That's it — build_index.py imports SUPPORTED_EXTENSIONS from here automatically.
"""

import os
import csv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth for supported file extensions.
# Imported by build_index.py — never duplicate this set elsewhere.
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {".csv", ".pdf", ".txt"}


def load_document(file_path: str) -> list[Document]:
    """
    Dispatch to the appropriate loader based on file extension.
    Returns a list of LangChain Document objects.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        return _load_csv(file_path)
    elif extension == ".pdf":
        return _load_pdf(file_path)
    elif extension == ".txt":
        return _load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def _load_csv(file_path: str) -> list[Document]:
    """
    Custom CSV loader that formats each bus route into a clean, searchable Document
    without exposing internal unique identifiers (Reg, Route) in page_content.
    """
    docs = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            bus_name = row.get("BusName", "").strip()
            bus_type = row.get("Type", "").strip()
            start = row.get("Start", "").strip()
            end = row.get("End", "").strip()
            seats = row.get("Seats", "").strip()
            seat_type = row.get("SeatType", "").strip()
            min_fare = row.get("MinFare", "").strip()
            add_stop_fare = row.get("AddStopFare", "").strip()
            facilities = row.get("Facilities", "").strip()
            stops = row.get("Stops", "").strip()
            departure = row.get("Departure", "").strip()
            arrival = row.get("Arrival", "").strip()
            travel_time = row.get("TravelTime", "").strip()

            content_lines = [
                f"Bus Name: {bus_name}",
                f"Bus Type: {bus_type}",
                f"Origin (Start): {start}",
                f"Destination (End): {end}",
                f"Departure Time: {departure}",
                f"Arrival Time: {arrival}",
                f"Travel Duration: {travel_time}",
                f"Available Seats: {seats} ({seat_type})",
                f"Minimum Fare (First Stop): ₹{min_fare}",
                f"Additional Stop Fare (after first stop): ₹{add_stop_fare}",
                f"Facilities Onboard: {facilities}",
                f"Route Intermediate Stops: {stops}",
            ]
            page_content = "\n".join(content_lines)

            metadata = {
                "source": os.path.basename(file_path),
                "row": i,
                "bus_name": bus_name,
                "start": start,
                "end": end,
                "type": bus_type,
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def _load_pdf(file_path: str) -> list[Document]:
    """PyPDFLoader returns one Document per page."""
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()


def _load_txt(file_path: str) -> list[Document]:
    """TextLoader returns a single Document with the full file content."""
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    return loader.load()


def load_structured_routes(file_path: str) -> list[dict]:
    """
    Parses CSV files into clean dict records suitable for insertion
    into the `structured_bus_routes` collection in MongoDB.
    """
    if not file_path.endswith(".csv"):
        return []

    records = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            bus_name = row.get("BusName", "").strip()
            route_number = row.get("Route", f"R-{i+1}").strip() or f"R-{i+1}"
            start = row.get("Start", "").strip()
            end = row.get("End", "").strip()
            bus_type = row.get("Type", "").strip()
            
            raw_stops = [
                s.strip()
                for s in row.get("Stops", "").replace("->", ",").split(",")
                if s.strip()
            ]
            stops_list = [{"stop_name": start, "sequence_order": 1, "fare_from_origin": 0.0}]
            
            try:
                min_fare = float(row.get("MinFare", "0").strip() or 0)
                add_fare = float(row.get("AddStopFare", "0").strip() or 0)
            except ValueError:
                min_fare, add_fare = 0.0, 0.0

            for idx, s_name in enumerate(raw_stops, start=2):
                stops_list.append({
                    "stop_name": s_name,
                    "sequence_order": idx,
                    "fare_from_origin": min_fare + (idx - 2) * add_fare
                })
            
            if end and (not raw_stops or raw_stops[-1].lower() != end.lower()):
                stops_list.append({
                    "stop_name": end,
                    "sequence_order": len(stops_list) + 1,
                    "fare_from_origin": min_fare + len(raw_stops) * add_fare
                })

            record = {
                "route_id": f"ROUTE-{i+1:04d}",
                "route_number": route_number,
                "bus_name": bus_name,
                "bus_type": bus_type,
                "origin": start,
                "destination": end,
                "stops": stops_list,
                "operating_hours": {
                    "first_bus": row.get("Departure", "06:00").strip(),
                    "last_bus": row.get("Arrival", "22:00").strip(),
                    "frequency_mins": 30
                },
                "fare_structure": {
                    "base_fare": min_fare,
                    "per_km_rate": add_fare
                },
                "source_file": os.path.basename(file_path)
            }
            records.append(record)
    return records



