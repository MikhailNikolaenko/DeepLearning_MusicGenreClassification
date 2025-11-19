import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List, Optional, Tuple

def scrape_everynoise_playlist(genre: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Scrape the Spotify playlist ID for a given genre from everynoise.com
    
    Returns:
        tuple: (playlist_id, error_message)
    """
    url = f"https://www.everynoise.com/everynoise1d-{genre}.html"
    
    try:
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the Spotify link with id="spotifylink"
        spotify_link = soup.find('a', id='spotifylink')
        
        if not spotify_link:
            return None, "No Spotify link found"
        
        # Get the href attribute
        href = spotify_link.get('href')
        
        if not href or not href.startswith('spotify:playlist:'):
            return None, f"Invalid Spotify link format: {href}"
        
        # Extract playlist ID (clean it)
        playlist_id = href.replace('spotify:playlist:', '')
        
        return playlist_id, None
        
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP Error {e.response.status_code}"
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def scrape_genres(genres: List[str], delay: float = 1.0) -> Dict:
    """
    Scrape Spotify playlists for multiple genres
    
    Args:
        genres: List of genre names
        delay: Delay between requests in seconds (be respectful!)
    
    Returns:
        Dictionary with successful and failed genres
    """
    results = {
        "playlists": {},
        "failed": {}
    }
    
    print(f"Starting to scrape {len(genres)} genres...\n")
    
    for i, genre in enumerate(genres, 1):
        print(f"[{i}/{len(genres)}] Scraping '{genre}'...", end=" ")
        
        playlist_id, error = scrape_everynoise_playlist(genre)
        
        if playlist_id:
            # Construct the full Spotify URL
            spotify_url = f"https://open.spotify.com/playlist/{playlist_id}"
            results["playlists"][genre] = spotify_url
            print(f"Success: {playlist_id}")
        else:
            results["failed"][genre] = error
            print(f"Failed: {error}")
        
        # Be respectful - add delay between requests
        # if i < len(genres):
        #     time.sleep(delay)
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(results['playlists'])} successful, {len(results['failed'])} failed")
    print(f"{'='*60}\n")
    
    return results


def main():
    # List of genres to scrape
    genres = [
        "pop",
        "rock",
        "jazz",
        # "classical",
        # "hiphop",
        # "country",
        # "blues",
        # "reggae",
        # "metal",
        # "disco",
        # "electronic",
        # "folk",
        # "indie",
        # "punk",
        # "soul",
        # "funk",
        # "rnb",
        # "latin",
        # "ambient",
        # "techno"
    ]
    
    # Scrape all genres
    results = scrape_genres(genres, delay=0.0)
    
    # Save to JSON file
    output_file = "everynoise_playlists.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    
    # Print failed genres if any
    if results["failed"]:
        print("\nFailed genres:")
        for genre, error in results["failed"].items():
            print(f"  - {genre}: {error}")
    
    # Print successful playlists
    if results["playlists"]:
        print(f"\nSuccessfully scraped {len(results['playlists'])} playlists:")
        for genre, url in list(results["playlists"].items())[:5]:
            print(f"  - {genre}: {url}")
        if len(results["playlists"]) > 5:
            print(f"  ... and {len(results['playlists']) - 5} more")


if __name__ == "__main__":
    main()
