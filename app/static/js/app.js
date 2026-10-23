// ===================================================================
// SFIA Skill Mapping - Interactive Single Page Application Logic
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
  
  const alertBox = document.getElementById('alertBox');
  const alertText = document.getElementById('alertText');
  
  const loadingCard = document.getElementById('loadingCard');
  const loadingStatusText = document.getElementById('loadingStatusText');
  const stepItems = document.querySelectorAll('.stepper-item');
  
  const resultsContainer = document.getElementById('resultsContainer');
  
  // Charts instances
  let radarChartInstance = null;
  let levelChartInstance = null;
  let currentAnalysisData = null;

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

  // 2. Fetch and render Sample Jobs
  async function loadSampleJobs() {
    try {
      const res = await fetch('/api/sample-jobs');
      const data = await res.json();
      if (data.success && data.samples) {
        sampleJobsContainer.innerHTML = '';
        data.samples.forEach(sample => {
          const chip = document.createElement('button');
          chip.className = 'sample-chip';
          chip.textContent = sample.title;
          chip.addEventListener('click', () => {
            // Switch to manual tab and prefill
            document.querySelector('[data-tab="tabManual"]').click();
            manualTitleInput.value = sample.title;
            manualDescInput.value = sample.description;
            clusterSelect.value = sample.cluster;
            // Smooth scroll to analyze button
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

  // 3. Helper Alert functions
  function showAlert(message) {
    alertText.textContent = message;
    alertBox.style.display = 'flex';
    alertBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function hideAlert() {
    alertBox.style.display = 'none';
  }

  // 4. Stepper Progress Visualizer
  function resetStepper() {
    stepItems.forEach(item => {
      item.classList.remove('active', 'done');
    });
  }

  function activateStep(index, statusText) {
    if (loadingStatusText) loadingStatusText.textContent = statusText;
    stepItems.forEach((item, i) => {
      if (i < index) {
        item.classList.remove('active');
        item.classList.add('done');
      } else if (i === index) {
        item.classList.add('active');
        item.classList.remove('done');
      } else {
        item.classList.remove('active', 'done');
      }
    });
  }

  const modelSelect = document.getElementById('modelSelect');
  
  // 5. Scrape from URL and automatically run Analysis
  btnScrape.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) {
      showAlert('Please enter a valid job URL from LinkedIn or JobStreet.');
      return;
    }

    hideAlert();
    resultsContainer.style.display = 'none';
    loadingCard.style.display = 'block';
    resetStepper();
    loadingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

    try {
      activateStep(0, `Scraping job description from ${url.includes('linkedin') ? 'LinkedIn' : url.includes('jobstreet') ? 'JobStreet' : 'Job Portal'}...`);
      
      const scrapeRes = await fetch('/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const scrapeData = await scrapeRes.json();

      if (!scrapeData.success) {
        loadingCard.style.display = 'none';
        showAlert(scrapeData.error || 'Failed to scrape job details.');
        // Prefill manual tab as fallback
        document.querySelector('[data-tab="tabManual"]').click();
        return;
      }

      // Prefill manual form behind the scenes
      manualTitleInput.value = scrapeData.title || 'Scraped Job';
      manualDescInput.value = scrapeData.description || '';

      const chosenModel = modelSelect ? modelSelect.value : 'skillner_qe_tfidf';

      // Trigger analysis directly
      await executeAnalysis({
        job_title: scrapeData.title,
        job_description: scrapeData.description,
        cluster: clusterSelect.value,
        threshold: parseFloat(thresholdInput.value) || 0.20,
        model_type: chosenModel,
        company: scrapeData.company,
        source: scrapeData.source
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
    resetStepper();
    loadingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

    await executeAnalysis({
      job_title: title,
      job_description: description,
      cluster: cluster,
      threshold: threshold,
      model_type: chosenModel,
      company: 'Direct Input',
      source: 'Manual'
    });
  });

  // 7. Core Execution Function
  async function executeAnalysis(payload) {
    try {
      activateStep(1, 'Preprocessing & Translating text to English...');
      await new Promise(r => setTimeout(r, 350));

      activateStep(2, 'Extracting Skills with SkillNER & Cleaned WordNet QE...');
      await new Promise(r => setTimeout(r, 400));

      if (payload.model_type && payload.model_type.includes('keybert')) {
        activateStep(3, 'Filtering and Ranking keyphrases via KeyBERT (MMR)...');
      } else {
        activateStep(3, 'Preparing Technical Skill Vector Representations...');
      }

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

      activateStep(4, 'Performing TF-IDF Cosine Similarity & SFIA Level Expansion...');
      await new Promise(r => setTimeout(r, 350));

      // Done
      stepItems.forEach(item => item.classList.add('done'));
      loadingStatusText.textContent = 'Mapping Complete!';
      await new Promise(r => setTimeout(r, 250));

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

    // Counters
    document.getElementById('statSkillsExtracted').textContent = res.counts.expanded_skills_count || res.counts.skillner_count;
    document.getElementById('statMappedCompetencies').textContent = res.counts.unique_skills_mapped;
    document.getElementById('statExpandedLevels').textContent = res.counts.expanded_competencies_count;

    // Extracted Skills Pills
    const keybertPillsWrapper = document.getElementById('keybertPillsWrapper');
    const keybertGroupTitle = document.querySelector('.pills-group-title');
    keybertPillsWrapper.innerHTML = '';
    const keybertSkills = res.extracted_skills.keybert || [];
    
    if (keybertSkills.length > 0) {
      if (keybertGroupTitle) keybertGroupTitle.textContent = 'KeyBERT Refined & Filtered Core Keyphrases:';
      keybertSkills.forEach(item => {
        const pill = document.createElement('span');
        pill.className = 'skill-pill keybert';
        pill.innerHTML = `<span>${item.keyword}</span><span class="score-badge">${(item.score * 100).toFixed(0)}%</span>`;
        keybertPillsWrapper.appendChild(pill);
      });
    } else {
      if (keybertGroupTitle) keybertGroupTitle.textContent = 'Cleaned WordNet Expanded Skill Terms (No Slang):';
      const qeSkills = (res.extracted_skills.wordnet_qe || []).filter(s => !res.extracted_skills.skillner.includes(s));
      if (qeSkills.length === 0) {
        keybertPillsWrapper.innerHTML = '<span style="color: var(--text-dim); font-size: 0.9rem;">Clean dictionary exact entities used directly.</span>';
      } else {
        qeSkills.slice(0, 20).forEach(term => {
          const pill = document.createElement('span');
          pill.className = 'skill-pill keybert';
          pill.style.background = 'rgba(79, 172, 254, 0.15)';
          pill.style.borderColor = 'rgba(79, 172, 254, 0.35)';
          pill.style.color = '#7dd3fc';
          pill.textContent = term;
          keybertPillsWrapper.appendChild(pill);
        });
      }
    }

    // Raw SkillNER pills
    const rawPillsWrapper = document.getElementById('rawPillsWrapper');
    rawPillsWrapper.innerHTML = '';
    const rawSkills = res.extracted_skills.skillner || [];
    rawSkills.slice(0, 20).forEach(skill => {
      const pill = document.createElement('span');
      pill.className = 'skill-pill raw';
      pill.textContent = skill;
      rawPillsWrapper.appendChild(pill);
    });
    if (rawSkills.length > 20) {
      const morePill = document.createElement('span');
      morePill.className = 'skill-pill raw';
      morePill.textContent = `+${rawSkills.length - 20} more`;
      rawPillsWrapper.appendChild(morePill);
    }

    // Render Charts
    renderRadarChart(res.radar_data);
    renderLevelDistributionChart(res.level_distribution);

    // Render SFIA Competencies Cards
    renderSFIACompetencies(res.mapped_skills);

    // Reveal results container with smooth animation
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // 9. Render SFIA Radar Chart (6 Categories)
  function renderRadarChart(radarData) {
    const ctx = document.getElementById('sfiaRadarChart').getContext('2d');
    if (radarChartInstance) {
      radarChartInstance.destroy();
    }

    radarChartInstance = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: radarData.labels.map(l => l.length > 18 ? l.substring(0, 18) + '...' : l),
        datasets: [{
          label: 'Competency Concentration',
          data: radarData.counts,
          backgroundColor: 'rgba(0, 242, 254, 0.22)',
          borderColor: '#00f2fe',
          pointBackgroundColor: '#4facfe',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#00f2fe',
          borderWidth: 2,
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            pointLabels: {
              color: '#94a3b8',
              font: { family: 'Outfit', size: 11, weight: '600' }
            },
            ticks: {
              backdropColor: 'transparent',
              color: '#64748b',
              stepSize: 1,
              font: { family: 'JetBrains Mono', size: 9 }
            }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // 10. Render Level Distribution Bar Chart
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
            'rgba(79, 172, 254, 0.4)',
            'rgba(79, 172, 254, 0.5)',
            'rgba(0, 242, 254, 0.6)',
            'rgba(0, 242, 254, 0.75)',
            'rgba(16, 185, 129, 0.7)',
            'rgba(138, 43, 226, 0.7)',
            'rgba(244, 63, 94, 0.7)'
          ],
          borderColor: '#00f2fe',
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { color: '#64748b', stepSize: 1, font: { family: 'JetBrains Mono', size: 10 } }
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
      container.innerHTML = '<div class="glass-card" style="text-align: center; color: var(--text-dim); padding: 2rem;">No SFIA competencies matched with the current threshold. Try lowering the similarity threshold.</div>';
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

      // Target level description (cleaned from Excel artifacts)
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
        <div style="font-size: 0.78rem; color: var(--text-dim); margin-top: 0.6rem; font-weight: 600;">MATCHED SIGNALS:</div>
        <div class="tokens-row">${tokensHtml}</div>
        ` : ''}

        <button class="desc-toggle-btn" data-target="desc-${idx}">
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
      btn.addEventListener('click', (e) => {
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
