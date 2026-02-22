# Migration from Freebase to Wikidata

This document describes the changes made to migrate HydraRAG from using Freebase to using Wikidata exclusively through the public API.

## Overview

HydraRAG originally supported both Freebase and Wikidata knowledge graphs. With this migration, the system now uses Wikidata only through its public API, removing the dependency on local Freebase services.

## Changes Made

### 1. Core Dependencies
- Removed import of `freebase_func` module
- Added import of `wikidata_api_client` module
- Created new Wikidata API client using Wikidata SPARQL endpoint

### 2. Function Replacement
- Replaced Freebase-specific functions with Wikidata equivalents:
  - `execurte_sparql()` → Placeholder function (deprecated)
  - `execute_sparql()` → Placeholder function (deprecated)
  - `replace_relation_prefix()` → Compatible wrapper
  - `replace_entities_prefix()` → Compatible wrapper
  - `id2entity_name_or_type()` → Now uses Wikidata API

### 3. Hydra Main Updates
- Modified `main_rag_process_multi()` to remove Freebase threading
- Updated argument parser to remove `--no-freebase` option
- Changed default `using_freebase` to `False`
- Updated database paths to local Wikidata paths
- Modified path selection to exclude "freebaseKG" source

### 4. Source Selection
- Updated `select_source_agent()` function to work with Wikidata only
- Modified logic to not default to enabling Freebase

## New Wikidata API Client Features

The new `wikidata_api_client.py` provides:
- Public SPARQL endpoint access to Wikidata
- Entity label lookup
- Relation discovery (incoming/outgoing)
- Entity search by name
- Tail/head entity resolution

## Configuration

The system now operates with only these knowledge sources:
- Wikidata Knowledge Graph (`--no-wikikg` to disable)
- Wikipedia Documents (`--no-wikidocu` to disable)
- Web Search (`--no-web` to disable)

Freebase support has been completely removed.

## Database Schema Changes

- Local database paths changed from `/data1/xingyut/.../free_subgraph/` to `../wikidata_subgraph/`
- Freebase-related tracking fields set to 0 by default
- Path source identification updated to recognize "wikiKG" instead of "freebaseKG"

## Compatibility Notes

- Existing graph loading functions remain compatible as they work with generic graph structures
- The system maintains the same interface for path generation and retrieval
- Only the data source has changed from Freebase to Wikidata