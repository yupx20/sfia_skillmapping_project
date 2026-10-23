import re
import json
import logging
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def clean_html_text(html_content: str) -> str:
    """Extract clean readable text from HTML markup."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style", "noscript", "svg", "button"]):
        script_or_style.decompose()
    
    # Replace list items with bulleted text
    for li in soup.find_all("li"):
        li.insert_before("\n• ")
    for br in soup.find_all(["br", "p", "div", "h1", "h2", "h3", "h4"]):
        br.insert_before("\n")

    text = soup.get_text()
    # Normalize multiple newlines and spaces
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(chunk for chunk in lines if chunk)
    return clean_text

def extract_linkedin_job_id(url: str) -> Optional[str]:
    """Extract job ID from various LinkedIn URL formats."""
    # Pattern 1: /jobs/view/4249179077 or /jobs/view/4249179077/
    match = re.search(r"/jobs/view/(\d+)", url)
    if match:
        return match.group(1)
    
    # Pattern 2: currentJobId=4249179077
    match = re.search(r"currentJobId=(\d+)", url)
    if match:
        return match.group(1)
    
    # Pattern 3: jobId in query parameter
    match = re.search(r"[?&]jobId=(\d+)", url)
    if match:
        return match.group(1)
    
    # Pattern 4: any 9-11 digit sequence
    digits = re.findall(r"\b\d{9,11}\b", url)
    if digits:
        return digits[0]
        
    return None

def scrape_linkedin_job(url: str) -> Dict[str, Any]:
    """Scrape job details from LinkedIn using public guest job endpoint."""
    job_id = extract_linkedin_job_id(url)
    
    # Attempt 1: LinkedIn Public Guest Job Endpoint (cleanest and doesn't require session)
    if job_id:
        guest_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        try:
            resp = requests.get(guest_url, headers=DEFAULT_HEADERS, timeout=12)
            if resp.status_code == 200 and resp.text:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                title_elem = soup.find(["h2", "h1"], class_=re.compile(r"top-card-layout__title|topcard__title|title"))
                company_elem = soup.find(["a", "span"], class_=re.compile(r"topcard__org-name-link|topcard__flavor"))
                location_elem = soup.find("span", class_=re.compile(r"topcard__flavor--bullet"))
                desc_elem = soup.find("div", class_=re.compile(r"show-more-less-html__markup|description__text|decoratedJobPosting"))
                
                job_title = title_elem.get_text(strip=True) if title_elem else "Job Title (LinkedIn)"
                company = company_elem.get_text(strip=True) if company_elem else "Unknown Company"
                location = location_elem.get_text(strip=True) if location_elem else ""
                
                if desc_elem:
                    description = clean_html_text(str(desc_elem))
                    if len(description) > 50:
                        return {
                            "success": True,
                            "source": "LinkedIn",
                            "job_id": job_id,
                            "title": job_title,
                            "company": company,
                            "location": location,
                            "description": description,
                            "url": url,
                        }
        except Exception as e:
            logger.warning(f"Guest API request failed for LinkedIn job {job_id}: {e}")

    # Attempt 2: Direct URL fetch
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            
            title_elem = soup.find(["h1", "h2"], class_=re.compile(r"title|job-title"))
            company_elem = soup.find(["a", "span"], class_=re.compile(r"company|organization"))
            desc_elem = soup.find("div", class_=re.compile(r"description|show-more-less|details"))
            
            if desc_elem:
                description = clean_html_text(str(desc_elem))
                if len(description) > 50:
                    return {
                        "success": True,
                        "source": "LinkedIn",
                        "job_id": job_id or "",
                        "title": title_elem.get_text(strip=True) if title_elem else "Job Title (LinkedIn)",
                        "company": company_elem.get_text(strip=True) if company_elem else "Company",
                        "location": "",
                        "description": description,
                        "url": url,
                    }
    except Exception as e:
        logger.warning(f"Direct fetch failed for LinkedIn: {e}")

    return {
        "success": False,
        "source": "LinkedIn",
        "error": (
            "Could not automatically scrape this LinkedIn link. LinkedIn often restricts public access "
            "or requires account login. Please copy the job description text and paste it into the 'Manual Input' tab."
        ),
        "url": url,
    }

def extract_jobstreet_job_id(url: str) -> Optional[str]:
    """Extract job ID from JobStreet URL."""
    match = re.search(r"/job/(\d+)", url)
    if match:
        return match.group(1)
    match = re.search(r"jobId=(\d+)", url)
    if match:
        return match.group(1)
    digits = re.findall(r"\b\d{7,10}\b", url)
    if digits:
        return digits[0]
    return None

def scrape_jobstreet_job(url: str) -> Dict[str, Any]:
    """Scrape job details from JobStreet."""
    job_id = extract_jobstreet_job_id(url)
    
    # Attempt 1: Direct HTML page fetch
    headers = dict(DEFAULT_HEADERS)
    headers["Host"] = "id.jobstreet.com" if "id.jobstreet.com" in url else "www.jobstreet.co.id"
    
    try:
        fetch_url = f"https://id.jobstreet.com/id/job/{job_id}" if job_id else url
        resp = requests.get(fetch_url, headers=DEFAULT_HEADERS, timeout=12)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # JobStreet data-automation attributes
            title_elem = soup.find("h1", {"data-automation": "job-detail-title"}) or soup.find("h1")
            company_elem = soup.find(["span", "a"], {"data-automation": "advertiser-name"})
            desc_elem = soup.find("div", {"data-automation": "jobAdDetails"}) or soup.find("div", class_=re.compile(r"jobAdDetails|jobDetails"))
            
            if desc_elem:
                description = clean_html_text(str(desc_elem))
                if len(description) > 50:
                    return {
                        "success": True,
                        "source": "JobStreet",
                        "job_id": job_id or "",
                        "title": title_elem.get_text(strip=True) if title_elem else "Job Title (JobStreet)",
                        "company": company_elem.get_text(strip=True) if company_elem else "Company",
                        "location": "",
                        "description": description,
                        "url": url,
                    }
    except Exception as e:
        logger.warning(f"Direct fetch failed for JobStreet: {e}")

    # Attempt 2: JobStreet internal search API
    if job_id:
        try:
            api_url = f"https://id.jobstreet.com/api/jobsearch/v5/job/{job_id}"
            resp = requests.get(api_url, headers=DEFAULT_HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", "")
                company = data.get("advertiser", {}).get("description", "")
                desc_html = data.get("jobAdDetails", "") or data.get("content", "")
                if desc_html:
                    return {
                        "success": True,
                        "source": "JobStreet",
                        "job_id": job_id,
                        "title": title or "Job Title (JobStreet)",
                        "company": company or "Company",
                        "location": "",
                        "description": clean_html_text(desc_html),
                        "url": url,
                    }
        except Exception as e:
            logger.warning(f"JobStreet API request failed for {job_id}: {e}")

    return {
        "success": False,
        "source": "JobStreet",
        "error": (
            "Could not automatically extract job content from this JobStreet link. "
            "Please copy and paste the job description directly into the 'Manual Input' tab."
        ),
        "url": url,
    }

def scrape_job_url(url: str) -> Dict[str, Any]:
    """Auto-detect source and scrape job posting from URL."""
    url = url.strip()
    if not url:
        return {"success": False, "error": "URL is empty."}
        
    if "linkedin.com" in url.lower():
        return scrape_linkedin_job(url)
    elif "jobstreet" in url.lower():
        return scrape_jobstreet_job(url)
    else:
        # Generic scrape attempt
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.find("h1")
                # find largest text container
                main_elem = soup.find(["main", "article"]) or soup.find("body")
                if main_elem:
                    description = clean_html_text(str(main_elem))
                    if len(description) > 100:
                        return {
                            "success": True,
                            "source": "Web",
                            "job_id": "",
                            "title": title.get_text(strip=True) if title else "Job Posting",
                            "company": "Company",
                            "location": "",
                            "description": description[:6000],
                            "url": url,
                        }
        except Exception as e:
            logger.warning(f"Generic scrape failed: {e}")
            
        return {
            "success": False,
            "error": "Unsupported job portal. Please provide a valid LinkedIn or JobStreet job posting link, or paste the text directly."
        }
