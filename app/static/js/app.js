// ===================================================================
// SFIA Skill Mapping - Minimalist Single Page Application Logic
// ===================================================================

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const urlInput = document.getElementById('jobUrlInput');
  const btnScrape = document.getElementById('btnScrape');
  const manualTitleInput = document.getElementById('manualTitleInput');
  const manualDescInput = document.getElementById('manualDescInput');
  const clusterSelect = document.getElementById('clusterSelect');
  const thresholdInput = document.getElementById('thresholdInput');
  const thresholdVal = document.getElementById('thresholdVal');
  const btnAnalyzeManual = document.getElementById('btnAnalyzeManual');
  const sampleJobsContainer = document.getElementById('sampleJobsContainer');
  const modelSelect = document.getElementById('modelSelect');
  
  const alertBox = document.getElementById('alertBox');
  const alertText = document.getElementById('alertText');
  
  const loadingCard = document.getElementById('loadingCard');
  const loadingStatusText = document.getElementById('loadingStatusText');
  const loadingPipelineText = document.getElementById('loadingPipelineText');
  const stepperList = document.getElementById('stepperList');
  
  const resultsContainer = document.getElementById('resultsContainer');
  
  // Chart instances
  let radarChartInstance = null;
  let levelChartInstance = null;
  let currentAnalysisData = null;

  // Pipeline Definitions Matching Backend Processes
  const MODEL_CONFIGS = {
    skillner_qe_tfidf: {
      name: 'SkillNER + QE + TF-IDF',
      subtitle: 'Pipeline: SkillNER, WordNet query expansion, and TF-IDF SFIA level mapping',
      steps: [
        { key: 'prep', label: 'Preprocessing', desc: 'Cleaning text and detecting language...' },
        { key: 'skillner', label: 'SkillNER Extraction', desc: 'Extracting technical competencies from taxonomy...' },
        { key: 'qe', label: 'Query Expansion', desc: 'Enriching technical terms via WordNet QE...' },
        { key: 'mapping', label: 'TF-IDF & Level Expansion', desc: 'Computing cosine similarity and expanding SFIA levels...' }
      ]
    },
    skillner_qe_keybert_tfidf: {
      name: 'SkillNER + Cleaned QE + KeyBERT + TF-IDF',
      subtitle: 'Pipeline: SkillNER, WordNet QE, KeyBERT MMR keyphrase ranking, and TF-IDF mapping',
      steps: [
        { key: 'prep', label: 'Preprocessing', desc: 'Cleaning text and detecting language...' },
        { key: 'skillner', label: 'SkillNER Extraction', desc: 'Extracting technical competencies from taxonomy...' },
        { key: 'qe', label: 'Query Expansion', desc: 'Enriching technical terms via WordNet QE...' },
        { key: 'keybert', label: 'KeyBERT MMR Ranking', desc: 'Ranking and filtering keyphrases via KeyBERT (MMR)...' },
        { key: 'mapping', label: 'TF-IDF & Level Expansion', desc: 'Computing cosine similarity and expanding SFIA levels...' }
      ]
    },
    skillner_tfidf: {
      name: 'SkillNER + TF-IDF',
      subtitle: 'Pipeline: Exact SkillNER dictionary taxonomy matching and TF-IDF level mapping',
      steps: [
        { key: 'prep', label: 'Preprocessing', desc: 'Cleaning text and detecting language...' },
        { key: 'skillner', label: 'SkillNER Exact Matching', desc: 'Extracting exact taxonomy entities (no expansion)...' },
        { key: 'mapping', label: 'TF-IDF & Level Expansion', desc: 'Computing cosine similarity and expanding SFIA levels...' }
      ]
    }
  };

  // Threshold Slider Listener
  if (thresholdInput && thresholdVal) {
    thresholdInput.addEventListener('input', (e) => {
      thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // 1. Tab Navigation
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      document.getElementById(targetTab).classList.add('active');
      hideAlert();
    });
  });

  // 2. Fetch and Render Sample Jobs
  async function loadSampleJobs() {
    try {
      const res = await fetch('/api/sample-jobs');
      const data = await res.json();
      if (data.success && data.samples) {
        sampleJobsContainer.innerHTML = '';
        data.samples.forEach(sample => {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'sample-chip';
          chip.textContent = sample.title;
          chip.addEventListener('click', () => {
            document.querySelector('[data-tab="tabManual"]').click();
            manualTitleInput.value = sample.title;
            manualDescInput.value = sample.description;
            clusterSelect.value = sample.cluster;
            manualDescInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
          });
          sampleJobsContainer.appendChild(chip);
        });
      }
    } catch (e) {
      console.warn('Could not load sample jobs:', e);
    }
  }
  loadSampleJobs();

  // 3. Alert Helpers
  function showAlert(message) {
    alertText.textContent = message;
    alertBox.style.display = 'flex';
    alertBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function hideAlert() {
    alertBox.style.display = 'none';
  }

  // 4. Dynamic Stepper Builder
  let activeStepElements = [];

  function buildStepper(modelKey, isScraping) {
    const config = MODEL_CONFIGS[modelKey] || MODEL_CONFIGS.skillner_qe_tfidf;
    loadingPipelineText.textContent = config.subtitle;
    
    // Construct steps list for current context
    const steps = [];
    if (isScraping) {
      steps.push({ key: 'scrape', label: 'URL Scraping', desc: 'Scraping job posting from source portal...' });
    }
    config.steps.forEach(s => steps.push(s));

    stepperList.innerHTML = '';
    activeStepElements = [];

    steps.forEach((step, index) => {
      const item = document.createElement('div');
      item.className = 'stepper-item';
      item.id = `dynStep-${index}`;

      const indicator = document.createElement('div');
      indicator.className = 'step-indicator';
      indicator.textContent = (index + 1).toString();

      const label = document.createElement('span');
      label.textContent = step.label;

      item.appendChild(indicator);
      item.appendChild(label);
      stepperList.appendChild(item);

      activeStepElements.push({ element: item, data: step });
    });

    return steps;
  }

  function activateStepIndex(index, customDesc) {
    activeStepElements.forEach((s, i) => {
      if (i < index) {
        s.element.classList.remove('active');
        s.element.classList.add('done');
      } else if (i === index) {
        s.element.classList.add('active');
        s.element.classList.remove('done');
        if (loadingStatusText) {
          loadingStatusText.textContent = customDesc || s.data.desc;
        }
      } else {
        s.element.classList.remove('active', 'done');
      }
    });
  }

  function completeAllSteps() {
    activeStepElements.forEach(s => {
      s.element.classList.remove('active');
      s.element.classList.add('done');
    });
    if (loadingStatusText) {
      loadingStatusText.textContent = 'Competency Mapping Complete';
    }
  }

  // 5. Scrape from URL and Automatically Run Analysis
  btnScrape.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) {
      showAlert('Please enter a valid job posting URL from LinkedIn or JobStreet.');
      return;
    }

    hideAlert();
    resultsContainer.style.display = 'none';
    loadingCard.style.display = 'block';
    loadingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

    const chosenModel = modelSelect ? modelSelect.value : 'skillner_qe_tfidf';
    buildStepper(chosenModel, true);

    try {
      activateStepIndex(0, `Retrieving job details from ${url.includes('linkedin') ? 'LinkedIn' : url.includes('jobstreet') ? 'JobStreet' : 'Job Portal'}...`);

      const scrapeRes = await fetch('/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const scrapeData = await scrapeRes.json();

      if (!scrapeData.success) {
        loadingCard.style.display = 'none';
        showAlert(scrapeData.error || 'Failed to scrape job details.');
        document.querySelector('[data-tab="tabManual"]').click();
        return;
      }

      // Prefill manual form
      manualTitleInput.value = scrapeData.title || 'Scraped Job';
      manualDescInput.value = scrapeData.description || '';

      // Proceed with model pipeline analysis
      await executeAnalysis({
        job_title: scrapeData.title,
        job_description: scrapeData.description,
        cluster: clusterSelect.value,
        threshold: parseFloat(thresholdInput.value) || 0.20,
        model_type: chosenModel,
        company: scrapeData.company,
        source: scrapeData.source,
        stepOffset: 1 // Stepper already had scraping at step 0
      });

    } catch (err) {
      loadingCard.style.display = 'none';
      showAlert(`An error occurred: ${err.message}`);
    }
  });

  // 6. Manual Analyze Button Click
  btnAnalyzeManual.addEventListener('click', async () => {
    const title = manualTitleInput.value.trim() || 'Untitled Job';
    const description = manualDescInput.value.trim();
    const cluster = clusterSelect.value;
    const threshold = parseFloat(thresholdInput.value) || 0.20;
    const chosenModel = modelSelect ? modelSelect.value : 'skillner_qe_tfidf';

    if (!description) {
      showAlert('Please provide a job description to extract and map skills.');
      return;
    }

    hideAlert();
    resultsContainer.style.display = 'none';
    loadingCard.style.display = 'block';
    loadingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

    buildStepper(chosenModel, false);

    await executeAnalysis({
      job_title: title,
      job_description: description,
      cluster: cluster,
      threshold: threshold,
      model_type: chosenModel,
      company: 'Direct Input',
      source: 'Manual',
      stepOffset: 0
    });
  });

  // 7. Core Execution Function with Model-Specific Step Timing
  async function executeAnalysis(payload) {
    const offset = payload.stepOffset || 0;
    const mType = payload.model_type || 'skillner_qe_tfidf';

    try {
      // Step: Preprocessing
      activateStepIndex(offset + 0, 'Normalizing text and translating language...');
      await new Promise(r => setTimeout(r, 220));

      // Step: SkillNER Extraction
      activateStepIndex(offset + 1, 'Extracting technical competencies using SkillNER taxonomy...');
      await new Promise(r => setTimeout(r, 260));

      if (mType === 'skillner_qe_keybert_tfidf') {
        // Step: WordNet QE
        activateStepIndex(offset + 2, 'Enriching technical synonyms and acronyms via WordNet QE...');
        await new Promise(r => setTimeout(r, 220));
        // Step: KeyBERT MMR
        activateStepIndex(offset + 3, 'Filtering and ranking keyphrases via KeyBERT (MMR)...');
      } else if (mType === 'skillner_qe_tfidf') {
        // Step: WordNet QE
        activateStepIndex(offset + 2, 'Enriching technical synonyms and acronyms via WordNet QE...');
      }

      // Execute actual backend request
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();

      if (!data.success) {
        loadingCard.style.display = 'none';
        showAlert(data.error || 'Skill analysis failed.');
        return;
      }

      // Final step: TF-IDF & Level Expansion
      const finalIndex = activeStepElements.length - 1;
      activateStepIndex(finalIndex, 'Computing cosine similarity and expanding SFIA levels...');
      await new Promise(r => setTimeout(r, 200));

      // Complete
      completeAllSteps();
      await new Promise(r => setTimeout(r, 220));

      loadingCard.style.display = 'none';
      renderResults(data.result, payload);

    } catch (err) {
      loadingCard.style.display = 'none';
      showAlert(`Inference failed: ${err.message}`);
    }
  }

  // 8. Render All Results onto Single Page
  function renderResults(res, payload) {
    currentAnalysisData = { ...res, meta: payload };

    // Overview details
    document.getElementById('metaJobTitle').textContent = res.job_title || payload.job_title;
    document.getElementById('metaCompany').textContent = payload.company || 'Direct Input';
    document.getElementById('metaCluster').textContent = `Cluster: ${res.detected_cluster} • ${res.active_model || 'SkillNER + QE + TF-IDF'}`;
    
    // Translation indicator
    const transIndicator = document.getElementById('translationBadge');
    if (transIndicator) {
      transIndicator.style.display = res.was_translated ? 'inline-block' : 'none';
    }

    // Execution Timings Display
    const timingsEl = document.getElementById('executionTimings');
    if (timingsEl && res.timings) {
      const t = res.timings;
      const timingParts = [];
      if (t.cleaning_translation) timingParts.push(`Preprocess: ${t.cleaning_translation}s`);
      if (t.skillner) timingParts.push(`SkillNER: ${t.skillner}s`);
      if (t.query_expansion) timingParts.push(`QE: ${t.query_expansion}s`);
      if (t.keybert) timingParts.push(`KeyBERT: ${t.keybert}s`);
      if (t.tfidf_cosine) timingParts.push(`TF-IDF: ${t.tfidf_cosine}s`);
      if (t.level_expansion) timingParts.push(`Level Expansion: ${t.level_expansion}s`);
      
      timingsEl.textContent = `Execution Telemetry: Total ${res.total_execution_time || 0}s (${timingParts.join(' • ')})`;
    }

    // Counters
    document.getElementById('statSkillsExtracted').textContent = res.counts.expanded_skills_count || res.counts.skillner_count;
    document.getElementById('statMappedCompetencies').textContent = res.counts.unique_skills_mapped;
    document.getElementById('statExpandedLevels').textContent = res.counts.expanded_competencies_count;

    // Extracted Skills Pills
    const keybertPillsWrapper = document.getElementById('keybertPillsWrapper');
    const primaryPillsTitle = document.getElementById('primaryPillsTitle');
    keybertPillsWrapper.innerHTML = '';
    const keybertSkills = res.extracted_skills.keybert || [];
    
    if (keybertSkills.length > 0) {
      if (primaryPillsTitle) primaryPillsTitle.textContent = 'KeyBERT Refined Keyphrases (MMR Relevance):';
      keybertSkills.forEach(item => {
        const pill = document.createElement('span');
        pill.className = 'skill-pill keybert';
        pill.innerHTML = `<span>${item.keyword}</span><span class="score-badge">${(item.score * 100).toFixed(0)}%</span>`;
        keybertPillsWrapper.appendChild(pill);
      });
    } else {
      if (primaryPillsTitle) primaryPillsTitle.textContent = 'WordNet Expanded Technical Terms:';
      const qeSkills = (res.extracted_skills.wordnet_qe || []).filter(s => !res.extracted_skills.skillner.includes(s));
      if (qeSkills.length === 0) {
        keybertPillsWrapper.innerHTML = '<span style="color: var(--text-dim); font-size: 0.85rem;">Clean exact taxonomy entities utilized directly.</span>';
      } else {
        qeSkills.slice(0, 24).forEach(term => {
          const pill = document.createElement('span');
          pill.className = 'skill-pill keybert';
          pill.textContent = term;
          keybertPillsWrapper.appendChild(pill);
        });
      }
    }

    // SkillNER direct taxonomy pills
    const rawPillsWrapper = document.getElementById('rawPillsWrapper');
    rawPillsWrapper.innerHTML = '';
    const rawSkills = res.extracted_skills.skillner || [];
    rawSkills.slice(0, 24).forEach(skill => {
      const pill = document.createElement('span');
      pill.className = 'skill-pill raw';
      pill.textContent = skill;
      rawPillsWrapper.appendChild(pill);
    });
    if (rawSkills.length > 24) {
      const morePill = document.createElement('span');
      morePill.className = 'skill-pill raw';
      morePill.textContent = `+${rawSkills.length - 24} more`;
      rawPillsWrapper.appendChild(morePill);
    }

    // Render Charts
    renderRadarChart(res.radar_data);
    renderLevelDistributionChart(res.level_distribution);

    // Render SFIA Competencies Cards
    renderSFIACompetencies(res.mapped_skills);

    // Reveal results container
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // 9. Render SFIA Radar Chart (6 Categories) - Minimalist Gold Theme
  function renderRadarChart(radarData) {
    const ctx = document.getElementById('sfiaRadarChart').getContext('2d');
    if (radarChartInstance) {
      radarChartInstance.destroy();
    }

    radarChartInstance = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: radarData.labels.map(l => l.length > 20 ? l.substring(0, 20) + '...' : l),
        datasets: [{
          label: 'Competency Concentration',
          data: radarData.counts,
          backgroundColor: 'rgba(217, 119, 6, 0.16)',
          borderColor: '#B48216',
          pointBackgroundColor: '#B48216',
          pointBorderColor: '#FFFFFF',
          pointHoverBackgroundColor: '#FFFFFF',
          pointHoverBorderColor: '#92400E',
          borderWidth: 2,
          pointRadius: 4,
          pointBorderWidth: 1.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(24, 24, 27, 0.08)' },
            grid: { color: 'rgba(24, 24, 27, 0.08)' },
            pointLabels: {
              color: '#3F3F46',
              font: { family: "'Plus Jakarta Sans', sans-serif", size: 10.5, weight: '600' }
            },
            ticks: {
              backdropColor: 'transparent',
              color: '#71717A',
              stepSize: 1,
              font: { family: "'JetBrains Mono', monospace", size: 9 }
            }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // 10. Render Level Distribution Bar Chart - Minimalist Warm Progression
  function renderLevelDistributionChart(levelDist) {
    const ctx = document.getElementById('levelDistChart').getContext('2d');
    if (levelChartInstance) {
      levelChartInstance.destroy();
    }

    const labels = Object.keys(levelDist);
    const dataVals = Object.values(levelDist);

    levelChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Skill Proficiency Levels',
          data: dataVals,
          backgroundColor: [
            'rgba(217, 119, 6, 0.25)',
            'rgba(217, 119, 6, 0.38)',
            'rgba(217, 119, 6, 0.50)',
            'rgba(217, 119, 6, 0.65)',
            'rgba(180, 130, 22, 0.80)',
            'rgba(180, 83, 9, 0.88)',
            'rgba(146, 64, 14, 0.95)'
          ],
          borderColor: '#B48216',
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#52525B', font: { family: "'JetBrains Mono', monospace", size: 10 } }
          },
          y: {
            grid: { color: 'rgba(24, 24, 27, 0.05)' },
            ticks: { color: '#71717A', stepSize: 1, font: { family: "'JetBrains Mono', monospace", size: 10 } }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // 11. Render SFIA Competencies List
  function renderSFIACompetencies(skills) {
    const container = document.getElementById('competenciesList');
    container.innerHTML = '';

    if (!skills || skills.length === 0) {
      container.innerHTML = '<div class="card" style="text-align: center; color: var(--text-muted); padding: 2rem;">No SFIA competencies matched with the current threshold. Consider lowering the similarity threshold.</div>';
      return;
    }

    skills.forEach((skillData, idx) => {
      const card = document.createElement('div');
      card.className = 'competency-card';

      const simPct = (skillData.highest_score * 100).toFixed(1);
      const targetLvl = skillData.highest_level;

      // Construct Level Hierarchy Nodes (1..7)
      let levelsHtml = '';
      for (let l = 1; l <= 7; l++) {
        let nodeClass = 'level-node';
        if (l === targetLvl) {
          nodeClass += ' target';
        } else if (skillData.expanded_levels.includes(l)) {
          nodeClass += ' expanded';
        }
        levelsHtml += `<span class="${nodeClass}">L${l}</span>`;
      }

      // Matched tokens tags
      let tokensHtml = '';
      if (skillData.matched_tokens && skillData.matched_tokens.length > 0) {
        tokensHtml = skillData.matched_tokens.map(t => `<span class="token-tag">${t}</span>`).join('');
      }

      // Target level description (cleaned of Excel OpenXML artifacts)
      const rawDesc = skillData.descriptions[targetLvl] || 'Proficiency level description from SFIA 9.';
      const targetDesc = rawDesc
        .replace(/_x000D_[\n\r]*/g, '\n')
        .replace(/_x000D_/g, ' ')
        .trim();

      card.innerHTML = `
        <div class="card-top-row">
          <div>
            <h3 class="competency-name">${skillData.skill}</h3>
            <div class="competency-cat">${skillData.category} &bull; ${skillData.subcategory || 'General'}</div>
          </div>
          <div class="score-meter">
            <span class="similarity-val">${skillData.highest_score.toFixed(3)}</span>
            <div class="meter-bar">
              <div class="meter-fill" style="width: ${Math.min(100, Math.max(10, simPct))}%;"></div>
            </div>
          </div>
        </div>

        <div class="levels-hierarchy-bar">
          <span class="levels-label">SFIA LEVEL HIERARCHY:</span>
          ${levelsHtml}
        </div>

        ${tokensHtml ? `
        <div style="font-size: 0.74rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 600; text-transform: uppercase;">Matched Signals:</div>
        <div class="tokens-row">${tokensHtml}</div>
        ` : ''}

        <button class="desc-toggle-btn" data-target="desc-${idx}" type="button">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
          SFIA Level ${targetLvl} Description
        </button>
        <div id="desc-${idx}" class="level-desc-box">
          ${targetDesc}
        </div>
      `;

      container.appendChild(card);
    });

    // Attach Accordion Toggle listeners
    document.querySelectorAll('.desc-toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');
        const box = document.getElementById(targetId);
        if (box) {
          const isShown = box.style.display === 'block';
          box.style.display = isShown ? 'none' : 'block';
          btn.querySelector('svg').style.transform = isShown ? 'rotate(0deg)' : 'rotate(180deg)';
        }
      });
    });
  }

  // 12. Export Analysis to JSON
  const btnExportJson = document.getElementById('btnExportJson');
  if (btnExportJson) {
    btnExportJson.addEventListener('click', () => {
      if (!currentAnalysisData) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentAnalysisData, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `sfia_mapping_${currentAnalysisData.detected_cluster}_${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }
});
