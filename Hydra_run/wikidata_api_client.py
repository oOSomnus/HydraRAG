"""
Wikidata API Client that uses public Wikidata SPARQL endpoint
instead of local services.
"""
import requests
import json
from typing import Dict, List, Optional, Union
import time
from urllib.error import HTTPError


class WikidataAPIClient:
    """
    A client to interact with Wikidata using the public SPARQL endpoint
    """

    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql"):
        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HydraRAG/1.0 (contact@example.com)',
            'Accept': 'application/sparql-results+json',
        })

    def _execute_sparql(self, query: str) -> Optional[List[Dict]]:
        """
        Execute a SPARQL query against the Wikidata endpoint
        """
        attempts = 0
        while attempts < 3:
            try:
                response = self.session.get(
                    self.endpoint_url,
                    params={'query': query},
                    headers={'Accept': 'application/sparql-results+json'}
                )
                response.raise_for_status()

                results = response.json()
                return results["results"]["bindings"]
            except Exception as e:
                print(f"Wikidata API Error encountered. Retrying after 2 seconds... Error: {e}")
                time.sleep(2)
                attempts += 1

        print("Failed to execute query after multiple attempts.")
        return None

    def get_entity_labels(self, entity_id: str) -> List[str]:
        """
        Get labels for a Wikidata entity
        """
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT ?label WHERE {{
          wd:{entity_id} rdfs:label ?label .
          FILTER(LANG(?label) = "en")
        }}
        """
        results = self._execute_sparql(query)
        if results is None:
            return []

        return [result['label']['value'] for result in results]

    def get_outgoing_relations(self, entity_id: str) -> List[str]:
        """
        Get all outgoing relations (properties) from an entity
        """
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?property WHERE {{
          wd:{entity_id} ?property ?value .
          FILTER(ISIRI(?property))
          FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT 100
        """
        results = self._execute_sparql(query)
        if results is None:
            return []

        properties = []
        for result in results:
            prop_uri = result['property']['value']
            prop_id = prop_uri.replace('http://www.wikidata.org/prop/direct/', '')
            properties.append(prop_id)

        return properties

    def get_incoming_relations(self, entity_id: str) -> List[str]:
        """
        Get all incoming relations (properties) to an entity
        """
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?property WHERE {{
          ?value ?property wd:{entity_id} .
          FILTER(ISIRI(?property))
          FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT 100
        """
        results = self._execute_sparql(query)
        if results is None:
            return []

        properties = []
        for result in results:
            prop_uri = result['property']['value']
            prop_id = prop_uri.replace('http://www.wikidata.org/prop/direct/', '')
            properties.append(prop_id)

        return properties

    def get_tail_entities(self, entity_id: str, property_id: str) -> List[Dict[str, str]]:
        """
        Get tail entities given head entity and relation/property
        """
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?tailEntity ?tailLabel WHERE {{
          wd:{entity_id} wdt:{property_id} ?tailEntity .
          OPTIONAL {{ ?tailEntity rdfs:label ?tailLabel . FILTER(LANG(?tailLabel) = "en") }}
          FILTER(ISIRI(?tailEntity))
        }}
        LIMIT 100
        """
        results = self._execute_sparql(query)
        if results is None:
            return []

        entities = []
        for result in results:
            entity_info = {
                'id': result['tailEntity']['value'].replace('http://www.wikidata.org/entity/', ''),
                'label': result.get('tailLabel', {}).get('value', 'Unnamed Entity')
            }
            entities.append(entity_info)

        return entities

    def get_head_entities(self, entity_id: str, property_id: str) -> List[Dict[str, str]]:
        """
        Get head entities given tail entity and relation/property
        """
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT ?headEntity ?headLabel WHERE {{
          ?headEntity wdt:{property_id} wd:{entity_id} .
          OPTIONAL {{ ?headEntity rdfs:label ?headLabel . FILTER(LANG(?headLabel) = "en") }}
          FILTER(ISIRI(?headEntity))
        }}
        LIMIT 100
        """
        results = self._execute_sparql(query)
        if results is None:
            return []

        entities = []
        for result in results:
            entity_info = {
                'id': result['headEntity']['value'].replace('http://www.wikidata.org/entity/', ''),
                'label': result.get('headLabel', {}).get('value', 'Unnamed Entity')
            }
            entities.append(entity_info)

        return entities

    def search_entity_by_name(self, name: str) -> List[Dict[str, str]]:
        """
        Search for entities by name using Wikidata's search API
        """
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': name,
            'language': 'en',
            'format': 'json',
            'limit': 10
        }

        attempts = 0
        while attempts < 3:
            try:
                response = self.session.get(search_url, params=params)
                response.raise_for_status()

                data = response.json()
                if 'search' in data:
                    results = []
                    for item in data['search']:
                        results.append({
                            'id': item['id'],
                            'label': item.get('display', {}).get('label', {}).get('value', item['label']),
                            'description': item.get('description', 'No description')
                        })
                    return results
            except Exception as e:
                print(f"Wikidata search API Error encountered. Retrying after 2 seconds... Error: {e}")
                time.sleep(2)
                attempts += 1

        print("Failed to search entities after multiple attempts.")
        return []


# Singleton instance for the client
wikidata_client = WikidataAPIClient()


def get_entity_name_or_type(entity_id: str) -> str:
    """
    Wrapper function to get entity name or type similar to the original id2entity_name_or_type
    """
    labels = wikidata_client.get_entity_labels(entity_id)
    if labels:
        return labels[0]
    return "Unnamed Entity"


def get_outgoing_relations_for_entity(entity_id: str) -> List[str]:
    """
    Wrapper function to get outgoing relations
    """
    return wikidata_client.get_outgoing_relations(entity_id)


def get_incoming_relations_for_entity(entity_id: str) -> List[str]:
    """
    Wrapper function to get incoming relations
    """
    return wikidata_client.get_incoming_relations(entity_id)


def get_tail_entities_given_head_and_relation(entity_id: str, property_id: str) -> List[Dict[str, str]]:
    """
    Wrapper function to get tail entities
    """
    return wikidata_client.get_tail_entities(entity_id, property_id)


def get_head_entities_given_tail_and_relation(entity_id: str, property_id: str) -> List[Dict[str, str]]:
    """
    Wrapper function to get head entities
    """
    return wikidata_client.get_head_entities(entity_id, property_id)


if __name__ == "__main__":
    # Test the client
    client = WikidataAPIClient()

    # Test getting labels for a known entity
    print("Testing entity labels for Q42 (Douglas Adams):")
    labels = client.get_entity_labels("Q42")
    print(labels)

    # Test getting relations
    print("\nTesting outgoing relations for Q42:")
    relations = client.get_outgoing_relations("Q42")
    print(relations[:5])  # Print first 5

    # Test entity search
    print("\nTesting entity search for 'Albert Einstein':")
    entities = client.search_entity_by_name("Albert Einstein")
    for entity in entities[:3]:  # Print first 3
        print(f"ID: {entity['id']}, Label: {entity['label']}")


class WikidataClientAdapter:
    """
    Adapter class to make WikidataAPIClient compatible with the existing wiki_client interface
    that expects query_all, get_all_relations_of_an_entity, etc. methods.
    """
    def __init__(self):
        self.client = WikidataAPIClient()

    def query_all(self, method, *args):
        """Adapter method for the MultiServerWikidataQueryClient.query_all interface"""
        if method == "get_all_relations_of_an_entity":
            entity_id = args[0]
            return self._get_all_relations_of_an_entity(entity_id)
        elif method == "label2pid":
            label = args[0]
            return self._label2pid(label)
        elif method == "get_tail_entities_given_head_and_relation":
            head_entity_id, relation_id = args[0], args[1]
            return self._get_tail_entities_given_head_and_relation(head_entity_id, relation_id)
        elif method == "get_head_entities_given_tail_and_relation":
            tail_entity_id, relation_id = args[0], args[1]
            return self._get_head_entities_given_tail_and_relation(tail_entity_id, relation_id)
        elif method == "get_tail_values_given_head_and_relation":
            head_entity_id, relation_id = args[0], args[1]
            return self._get_tail_values_given_head_and_relation(head_entity_id, relation_id)
        elif method == "get_wikipedia_link":
            entity_id = args[0]
            return self._get_wikipedia_link(entity_id)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_all_relations_of_an_entity(self, entity_id: str):
        """Get all relations (both incoming and outgoing) for an entity"""
        # Get property labels
        outgoing_props = self.client.get_outgoing_relations(entity_id)
        incoming_props = self.client.get_incoming_relations(entity_id)

        # Fetch labels for properties
        head_relations = []
        for prop_id in outgoing_props[:50]:  # Limit to 50 for performance
            label = self._pid2label(prop_id)
            if label:
                head_relations.append({'label': label, 'qid': prop_id})

        tail_relations = []
        for prop_id in incoming_props[:50]:  # Limit to 50 for performance
            label = self._pid2label(prop_id)
            if label:
                tail_relations.append({'label': label, 'qid': prop_id})

        return {'head': head_relations, 'tail': tail_relations}

    def _label2pid(self, label: str):
        """Search for property ID by label"""
        # Use Wikidata search API to find property by label
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': label,
            'language': 'en',
            'format': 'json',
            'type': 'property',
            'limit': 1
        }
        try:
            response = self.client.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'search' in data and data['search']:
                return [data['search'][0]['id']]  # Return as list for compatibility
        except:
            pass
        return []  # Return empty list if not found

    def _pid2label(self, pid: str):
        """Get property label by ID"""
        # Map common property IDs to labels
        label_map = {
            'P31': 'instance of',
            'P279': 'subclass of',
            'P17': 'country',
            'P36': 'capital',
            'P57': 'director',
            'P161': 'cast member',
            'P136': 'genre',
            'P577': 'publication date',
            'P1545': 'series ordinal',
            'P166': 'award received',
            'P580': 'start time',
            'P582': 'end time',
            'P710': 'participant',
            'P123': 'publisher',
            'P159': 'headquarters location',
            'P495': 'country of origin',
        }
        if pid in label_map:
            return label_map[pid]

        # Try to fetch label from Wikidata
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?label WHERE {{
          wdt:{pid} rdfs:label ?label .
          FILTER(LANG(?label) = "en")
        }}
        """
        try:
            results = self.client._execute_sparql(query)
            if results and len(results) > 0:
                return results[0]['label']['value']
        except:
            pass

        return f"Property {pid}"

    def _get_tail_entities_given_head_and_relation(self, head_entity_id: str, relation_id: str):
        """Get tail entities given head entity and relation"""
        entities = self.client.get_tail_entities(head_entity_id, relation_id)
        tail = []
        for entity in entities:
            tail.append({'label': entity.get('label', 'Unknown'), 'qid': entity.get('id', '')})
        return {'tail': tail, 'head': []}

    def _get_head_entities_given_tail_and_relation(self, tail_entity_id: str, relation_id: str):
        """Get head entities given tail entity and relation"""
        entities = self.client.get_head_entities(tail_entity_id, relation_id)
        head = []
        for entity in entities:
            head.append({'label': entity.get('label', 'Unknown'), 'qid': entity.get('id', '')})
        return {'head': head, 'tail': []}

    def _get_tail_values_given_head_and_relation(self, head_entity_id: str, relation_id: str):
        """Get tail values (literal values) given head entity and relation"""
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?value WHERE {{
          wd:{head_entity_id} wdt:{relation_id} ?value .
          FILTER(!ISIRI(?value))
        }}
        LIMIT 100
        """
        results = self.client._execute_sparql(query)
        values = []
        if results:
            for result in results:
                val = result['value']['value']
                # Extract string values
                if '"' in val and '@' in val:
                    # Language-tagged string
                    val = val.split('"')[1]
                values.append(val)
        return list(set(values))

    def _get_wikipedia_link(self, entity_id: str):
        """Get Wikipedia link for entity"""
        # Get the English Wikipedia page
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX schema: <http://schema.org/>

        SELECT ?link WHERE {{
          wd:{entity_id} schema:about ?item .
          ?item schema:inLanguage "en" .
          ?item schema:isPartOf <https://en.wikipedia.org/> .
          BIND(STRAFTER(STR(?item), STR(<https://en.wikipedia.org/wiki/>)) AS ?link)
        }}
        LIMIT 1
        """
        results = self.client._execute_sparql(query)
        if results and len(results) > 0:
            return [results[0]['link']['value']]
        return []


# Singleton adapter instance for compatibility
wikidata_client_adapter = WikidataClientAdapter()