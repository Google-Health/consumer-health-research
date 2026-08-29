"""Web search tools for the Domain Expert Agent.

This module provides web search capabilities for retrieving authoritative
health information from the web using Tavily API or DuckDuckGo.
"""

import os
import requests
import urllib.parse
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single search result."""
    url: str
    title: str
    snippet: str
    score: Optional[float] = None


class WebSearchTool:
    """Web search tool supporting multiple backends.
    
    Supports:
    - Tavily API (recommended for health queries - has source filtering)
    - DuckDuckGo (free fallback, no API key needed)
    """
    
    def __init__(
        self,
        backend: str = "tavily",
        api_key: Optional[str] = None,
    ):
        """Initialize the web search tool.
        
        Args:
            backend: "tavily" or "duckduckgo"
            api_key: API key for Tavily (reads from TAVILY_API_KEY env var if not provided)
        
        Raises:
            ValueError: If backend is "tavily" but no API key is provided.
        """
        self.backend = backend
        
        if backend == "tavily":
            self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Tavily API key required. Either:\n"
                    "  1. Set TAVILY_API_KEY environment variable\n"
                    "  2. Pass api_key parameter\n"
                    "  3. Use backend='duckduckgo' for free (lower quality) search\n"
                    "Get a Tavily API key at: https://tavily.com/"
                )
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Perform a web search.
        
        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            include_domains: Optional list of domains to prefer (Tavily only).
        
        Returns:
            List of SearchResult objects.
        """
        if self.backend == "tavily":
            return self._search_tavily(query, max_results, include_domains)
        else:
            return self._search_duckduckgo(query, max_results)
    
    def _search_tavily(
        self,
        query: str,
        max_results: int,
        include_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search using Tavily API."""
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "Tavily package not installed. Run: pip install tavily-python"
            )
        
        client = TavilyClient(api_key=self.api_key)
        
        # Default to authoritative health domains if none specified
        if include_domains is None:
            include_domains = [
                "mayoclinic.org",
                "cdc.gov",
                "who.int",
                "nih.gov",
                "clevelandclinic.org",
                "webmd.com",
                "healthline.com",
            ]
        
        response = client.search(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
            search_depth="advanced",
        )
        
        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                score=item.get("score"),
            ))
        return results
    
    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search package required for fallback. "
                "Install with: pip install duckduckgo-search"
            )
        
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                    ))
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return results
    
    def search_formatted(
        self,
        query: str,
        max_results: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """Search and return results in simple format.
        
        Args:
            query: Search query string.
            max_results: Maximum number of results.
        
        Returns:
            Tuple of (list of URLs, list of snippets).
        """
        results = self.search(query, max_results)
        urls = [r.url for r in results]
        snippets = [r.snippet for r in results]
        return urls, snippets


class DataCommonsClient:
    """Client for querying Data Commons public statistics API.
    
    Data Commons provides access to public datasets from sources like
    CDC, WHO, Census Bureau, etc.
    """
    
    BASE_URL = "https://api.datacommons.org"
    
    def __init__(self):
        """Initialize the Data Commons client."""
        pass
    
    def query(self, query: str) -> dict:
        """Query Data Commons with natural language.
        
        Args:
            query: Natural language query about statistics.
        
        Returns:
            Dictionary with query results.
        """
        # Use the public NL query endpoint
        url = f"https://datacommons.org/api/explore/detect?q={urllib.parse.quote(query)}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_statistical_value(
        self,
        place: str,
        stat_var: str,
    ) -> Optional[float]:
        """Get a specific statistical value for a place.
        
        Args:
            place: Place DCID (e.g., "country/USA", "geoId/06")
            stat_var: Statistical variable (e.g., "Count_Person")
        
        Returns:
            The statistical value, or None if not found.
        """
        url = f"{self.BASE_URL}/v2/observation"
        params = {
            "entity.dcids": place,
            "select": ["entity", "variable", "value", "date"],
            "variable.dcids": stat_var,
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Extract the most recent value
                observations = data.get("byVariable", {}).get(stat_var, {}).get("byEntity", {}).get(place, {}).get("orderedFacets", [])
                if observations:
                    return observations[0].get("observations", [{}])[0].get("value")
            return None
        except Exception:
            return None


def get_web_search_tool(
    backend: str = "tavily",
    api_key: Optional[str] = None,
) -> WebSearchTool:
    """Factory function to create a web search tool.
    
    Args:
        backend: "tavily" or "duckduckgo"
        api_key: API key for Tavily
    
    Returns:
        Configured WebSearchTool instance.
    """
    return WebSearchTool(backend=backend, api_key=api_key)
