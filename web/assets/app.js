(function () {
  'use strict';

  var DATA = 'data/';
  var BLANK_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
  var C = { matched: '#5AA9DE', unmatched: '#F5A33E', structure: '#7C8A98', detection: '#5AA9DE' };
  var LABEL = {
    matched: 'Matched to AIS', unmatched: 'No AIS received',
    structure: 'Fixed structures', detection: 'Radar detections'
  };

  var state = {
    manifest: null, passes: [], idx: 0, features: [], selected: null,
    minLen: 0, minConf: 0,
    on: { matched: true, unmatched: true, structure: true, detection: true, footprint: false, overview: false }
  };

  var map, $ = function (id) { return document.getElementById(id); };

  function fetchJSON(path) {
    return fetch(DATA + path, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error(path + ': HTTP ' + r.status);
      return r.json();
    });
  }

  function fmtUTC(iso) {
    var d = new Date(iso);
    var mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getUTCMonth()];
    function p(n) { return (n < 10 ? '0' : '') + n; }
    return d.getUTCDate() + ' ' + mon + ' ' + d.getUTCFullYear() + ' · ' +
      p(d.getUTCHours()) + ':' + p(d.getUTCMinutes()) + ' UTC';
  }
  function fmtShort(iso) {
    var d = new Date(iso);
    return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getUTCMonth()] +
      ' ' + d.getUTCFullYear();
  }
  
  function squareIcon(px, rgb, stroke) {
    var d = new Uint8Array(px * px * 4);
    for (var y = 0; y < px; y++) {
      for (var x = 0; x < px; x++) {
        var edge = x < stroke || y < stroke || x >= px - stroke || y >= px - stroke;
        var i = (y * px + x) * 4;
        if (edge) { d[i] = rgb[0]; d[i + 1] = rgb[1]; d[i + 2] = rgb[2]; d[i + 3] = 255; }
      }
    }
    return { width: px, height: px, data: d };
  }

  function passesFilter(p) {
    if (p.cls === 'structure') return true;

    var len = p.len_m == null ? 0 : p.len_m;
    var conf = p.conf == null ? 1 : p.conf;
    return len >= state.minLen && conf >= state.minConf;
  }

  function visible() {
    return state.features.filter(function (f) {
      return state.on[f.properties.cls] && passesFilter(f.properties);
    });
  }

  function mlFilter(cls) {
    if (cls === 'structure') return ['==', ['get', 'cls'], 'structure'];

    return ['all',
      ['==', ['get', 'cls'], cls],
      ['>=', ['coalesce', ['get', 'len_m'], 0], state.minLen],
      ['>=', ['coalesce', ['get', 'conf'], 1], state.minConf]];
  }

  function buildMap(bbox) {
    map = new maplibregl.Map({
      container: 'map',
      style: { version: 8, sources: {}, layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#0E1620' } }] },
      bounds: [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
      fitBoundsOptions: { padding: 24 },
      maxBounds: [[bbox[0] - 3, bbox[1] - 3], [bbox[2] + 3, bbox[3] + 3]],
      attributionControl: false,
      dragRotate: false
    });
    map.touchZoomRotate.disableRotation();
    map.keyboard.enable();

    return new Promise(function (res) { map.on('load', res); });
  }

  function addLayers(man) {
    map.addSource('coast', { type: 'geojson', data: DATA + man.coastline.hi });
    map.addLayer({ id: 'land', type: 'fill', source: 'coast', paint: { 'fill-color': '#1E2731' } });
    map.addLayer({ id: 'shore', type: 'line', source: 'coast', paint: { 'line-color': '#33404D', 'line-width': 0.8 } });

    map.addSource('overview', { type: 'image', url: BLANK_PNG, coordinates: [[-1, 1], [1, 1], [1, -1], [-1, -1]] });
    map.addLayer({ id: 'overview', type: 'raster', source: 'overview', layout: { visibility: 'none' }, paint: { 'raster-opacity': 0.9 } }, 'land');

    map.addSource('footprint', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'footprint', type: 'line', source: 'footprint', layout: { visibility: 'none' }, paint: { 'line-color': '#5E6E7E', 'line-width': 1.2, 'line-dasharray': [5, 4] } });

    map.addSource('cells', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'cells', type: 'fill', source: 'cells', layout: { visibility: 'none' }, paint: { 'fill-color': '#3D7FB8', 'fill-opacity': 0.22, 'fill-outline-color': '#3D7FB8' } });

    map.addSource('det', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addImage('sq', squareIcon(20, [124, 138, 152], 3), { pixelRatio: 2 });

    map.addLayer({
      id: 'structure', type: 'symbol', source: 'det', filter: mlFilter('structure'),
      layout: { 'icon-image': 'sq', 'icon-allow-overlap': true }
    });
    map.addLayer({
      id: 'matched', type: 'circle', source: 'det', filter: mlFilter('matched'),
      paint: { 'circle-radius': 4.5, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': C.matched, 'circle-stroke-width': 1.8 }
    });
    map.addLayer({
      id: 'detection', type: 'circle', source: 'det', filter: mlFilter('detection'),
      paint: { 'circle-radius': 4.5, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': C.detection, 'circle-stroke-width': 1.8 }
    });
    map.addLayer({
      id: 'unmatched', type: 'circle', source: 'det', filter: mlFilter('unmatched'),
      paint: { 'circle-radius': 4.5, 'circle-color': C.unmatched }
    });

    map.addSource('sel', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'sel', type: 'circle', source: 'sel', paint: { 'circle-radius': 13, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#E6ECF2', 'circle-stroke-width': 1.4, 'circle-stroke-opacity': 0.8 } });

    ['matched', 'unmatched', 'detection', 'structure'].forEach(function (id) {
      map.on('click', id, function (e) { select(e.features[0]); });
      map.on('mouseenter', id, function () { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', id, function () { map.getCanvas().style.cursor = ''; });
    });
    map.on('click', function (e) {
      var hits = map.queryRenderedFeatures(e.point, { layers: ['matched', 'unmatched', 'detection', 'structure'] });
      if (!hits.length) closeDetail();
    });
    map.on('move', updateScale);
  }

  function updateScale() {
    var c = map.getCenter();
    var a = map.project(c), b = { x: a.x + 110, y: a.y };
    var m = new maplibregl.LngLat(c.lng, c.lat).distanceTo(map.unproject(b));

    var nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].reduce(function (best, k) {
      return Math.abs(k - m / 1000) < Math.abs(best - m / 1000) ? k : best;
    }, 1);
    $('scale-bar').style.width = Math.round(110 * (nice * 1000) / m) + 'px';
    $('scale-label').textContent = nice + ' km';
  }

  function buildLayerRows(man) {
    var rows = man.ais ? ['matched', 'unmatched', 'structure'] : ['detection', 'structure'];
    rows.push('footprint', 'overview');
    var host = $('layers');
    host.innerHTML = '';

    rows.forEach(function (key) {
      var lab = document.createElement('label');
      lab.className = 'layer';
      var swatch = '';

      if (key === 'matched' || key === 'detection')
        swatch = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="4.5" fill="none" stroke="' + C[key] + '" stroke-width="2"/></svg>';
      else if (key === 'unmatched')
        swatch = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="' + C.unmatched + '"/></svg>';
      else if (key === 'structure')
        swatch = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="3" y="3" width="8" height="8" fill="none" stroke="' + C.structure + '" stroke-width="1.6"/></svg>';
      else if (key === 'footprint')
        swatch = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="2.5" width="9" height="9" fill="none" stroke="#5E6E7E" stroke-width="1.4" stroke-dasharray="2 2"/></svg>';
      else
        swatch = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2" y="2" width="10" height="10" fill="#3D7FB8" opacity="0.55"/></svg>';
      
      var name = key === 'footprint' ? 'Scene footprint'
        : key === 'overview' ? 'Radar image' : LABEL[key];
      lab.innerHTML = '<input type="checkbox" ' + (state.on[key] ? 'checked' : '') + '>' + swatch +
        '<span class="name">' + name + '</span><span class="count" id="n-' + key + '"></span>';
      lab.querySelector('input').addEventListener('change', function (e) {
        state.on[key] = e.target.checked;
        lab.classList.toggle('off', !e.target.checked);
        applyVisibility();
        refreshCounts();
      });
      host.appendChild(lab);
    });
  }

  function applyVisibility() {
    ['matched', 'unmatched', 'detection', 'structure'].forEach(function (id) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', state.on[id] ? 'visible' : 'none');
    });

    map.setLayoutProperty('footprint', 'visibility', state.on.footprint ? 'visible' : 'none');
    var ov = state.passes[state.idx] && state.passes[state.idx].overview;
    map.setLayoutProperty('overview', 'visibility', (state.on.overview && ov) ? 'visible' : 'none');
  }

  function applyFilters() {
    ['matched', 'unmatched', 'detection', 'structure'].forEach(function (id) {
      if (map.getLayer(id)) map.setFilter(id, mlFilter(id));
    });
  }

  function refreshCounts() {
    var v = visible(), by = {};
    v.forEach(function (f) { by[f.properties.cls] = (by[f.properties.cls] || 0) + 1; });

    ['matched', 'unmatched', 'detection', 'structure'].forEach(function (k) {
      var el = $('n-' + k);
      if (el) el.textContent = by[k] || 0;
    });

    var pass = state.passes[state.idx] || {};
    var vessels = (by.matched || 0) + (by.unmatched || 0) + (by.detection || 0);
    $('s-total').textContent = v.length;
    $('s-agg').textContent = pass.aggregated_points == null ? '—' : pass.aggregated_points;

    if (state.manifest.ais) {
      $('s-pct').textContent = vessels ? Math.round((by.matched || 0) / vessels * 100) + '%' : '—';
      $('s-unm').textContent = by.unmatched || 0;
    } else {
      $('s-pct').textContent = by.detection || 0;
      $('s-pct-l').textContent = 'radar detections';
      $('s-unm').textContent = by.structure || 0;
      $('s-unm-l').textContent = 'fixed structures';
      $('s-unm').style.color = 'var(--structure)';
    }
    $('a11y-summary').textContent = v.length + ' detections shown for the pass of ' +
      (pass.acquired ? fmtUTC(pass.acquired) : 'unknown date') + '.';
  }

  function select(feature) {
    var p = feature.properties;
    if (typeof p.reasons === 'string') p.reasons = JSON.parse(p.reasons);
    if (typeof p.ais === 'string') p.ais = JSON.parse(p.ais);

    state.selected = p.id;
    var coords = feature.geometry.coordinates;
    map.getSource('sel').setData({ type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: coords } }] });

    $('d-kind').textContent = LABEL[p.cls] || p.cls;
    $('d-dot').setAttribute('fill', C[p.cls] || '#9CACBC');
    $('d-kind').className = 'k';
    $('d-kind').style.color = C[p.cls] || 'var(--text)';

    var chip = p.chip_vh || p.chip_vv;
    $('d-chip-wrap').hidden = !chip;
    if (chip) {
      $('d-chip').src = DATA + chip;
      $('d-chip-cap').textContent = (p.chip_vh ? 'VH' : 'VV') + ' · ' +
        (p.chip_px ? (p.chip_px * 10 / 1000).toFixed(1) + ' km' : '');
    }

    var rows = [
      ['Position', coords[1].toFixed(3) + '°N ' + coords[0].toFixed(3) + '°E'],
      ['Length', p.len_m != null ? Math.round(p.len_m) + ' m' : '—'],
      ['Confidence', p.conf != null ? Number(p.conf).toFixed(2) : '—']
    ];
    if (p.ais && p.ais.len_m != null) rows.push(['AIS length', Math.round(p.ais.len_m) + ' m']);
    if (p.ais && p.ais.sog != null) rows.push(['Speed', p.ais.sog + ' kn']);
    if (p.residual_m != null) rows.push(['Match residual', Math.round(p.residual_m) + ' m']);
    if (p.doppler_m != null) rows.push(['Doppler shift', Math.round(p.doppler_m) + ' m']);
    $('d-facts').innerHTML = rows.map(function (r) {
      return '<div><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>';
    }).join('');

    var reasons = p.reasons || [];
    $('d-reasons').innerHTML = reasons.map(function (r) {
      return '<li><span class="bullet k-' + (r.k || 'info') + '" aria-hidden="true">•</span><span>' +
        String(r.t).replace(/</g, '&lt;') + '</span></li>';
    }).join('') || '<li><span class="bullet k-info">•</span><span>No explanation recorded.</span></li>';

    $('detail').hidden = false;
    $('d-close').focus();
  }

  function closeDetail() {
    $('detail').hidden = true;
    state.selected = null;
    if (map.getSource('sel')) map.getSource('sel').setData({ type: 'FeatureCollection', features: [] });
  }

  function buildTrack() {
    var track = $('track');
    track.innerHTML = '';

    if (!state.passes.length) 
      return;

    var times = state.passes.map(function (p) { return new Date(p.acquired).getTime(); });
    var lo = Math.min.apply(null, times), hi = Math.max.apply(null, times);
    var span = Math.max(hi - lo, 1);

    state.passes.forEach(function (p, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'tick';
      b.style.left = ((times[i] - lo) / span * 100) + '%';
      b.setAttribute('aria-label', 'Pass of ' + fmtUTC(p.acquired));
      b.addEventListener('click', function () { loadPass(i); });
      track.appendChild(b);
    });

    $('p-range').textContent = fmtShort(state.passes[state.passes.length - 1].acquired) +
      ' – ' + fmtShort(state.passes[0].acquired);
  }

  function loadPass(i) {
    state.idx = i;
    var p = state.passes[i];
    closeDetail();

    $('p-when').textContent = fmtUTC(p.acquired);
    $('p-meta').textContent = (p.satellite || '') + (p.scenes ? ' · ' + p.scenes.length + ' scene' + (p.scenes.length > 1 ? 's' : '') : '');
    Array.prototype.forEach.call($('track').children, function (el, k) {
      el.setAttribute('aria-current', k === i ? 'true' : 'false');
    });
    $('p-prev').disabled = i >= state.passes.length - 1;
    $('p-next').disabled = i <= 0;

    fetchJSON(p.detections).then(function (fc) {
      state.features = fc.features || [];
      map.getSource('det').setData(fc);
      applyFilters();
      refreshCounts();
    }).catch(function (e) { console.error(e); });

    if (p.footprint) {
      fetchJSON(p.footprint).then(function (fc) { map.getSource('footprint').setData(fc); })
        .catch(function () { map.getSource('footprint').setData({ type: 'FeatureCollection', features: [] }); });
    } else {
      map.getSource('footprint').setData({ type: 'FeatureCollection', features: [] });
    }

    if (p.cells) {
      fetchJSON(p.cells).then(function (fc) {
        map.getSource('cells').setData(fc);
        map.setLayoutProperty('cells', 'visibility', 'visible');
      }).catch(function () {});
    } else {
      map.setLayoutProperty('cells', 'visibility', 'none');
    }

    if (p.overview) {
      var b = p.overview.bounds; // [[w,s], [e,n]]
      map.getSource('overview').updateImage({
        url: DATA + p.overview.url,
        coordinates: [[b[0][0], b[1][1]], [b[1][0], b[1][1]], [b[1][0], b[0][1]], [b[0][0], b[0][1]]]
      });
    }
    var ovRow = document.querySelector('#layers .layer:last-child');

    if (ovRow) {
      var ovInput = ovRow.querySelector('input');
      ovInput.disabled = !p.overview;
      ovRow.title = p.overview ? '' : 'No radar overview was exported for this pass';
      ovRow.classList.toggle('off', !p.overview || !state.on.overview);
    }
    applyVisibility();
  }

  function wire() {
    $('f-len').addEventListener('input', function (e) {
      state.minLen = Number(e.target.value);
      $('f-len-out').textContent = state.minLen + ' m';
      applyFilters(); refreshCounts();
    });
    $('f-conf').addEventListener('input', function (e) {
      state.minConf = Number(e.target.value) / 100;
      $('f-conf-out').textContent = state.minConf.toFixed(2);
      applyFilters(); refreshCounts();
    });
    $('d-close').addEventListener('click', closeDetail);
    $('z-in').addEventListener('click', function () { map.zoomIn(); });
    $('z-out').addEventListener('click', function () { map.zoomOut(); });
    $('z-reset').addEventListener('click', function () {
      var b = state.manifest.bbox;
      map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 24 });
    });
    $('p-prev').addEventListener('click', function () {
      if (state.idx < state.passes.length - 1) loadPass(state.idx + 1);
    });
    $('p-next').addEventListener('click', function () { if (state.idx > 0) loadPass(state.idx - 1); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeDetail(); });
  }

  function fail(msg) {
    document.querySelector('.map-wrap').insertAdjacentHTML('afterbegin',
      '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;' +
      'padding:40px;text-align:center;color:var(--text-2);z-index:5">' +
      '<div><p style="font-size:15px;margin:0 0 8px;color:var(--text)">No site data yet</p>' +
      '<p style="font-size:13px;margin:0;max-width:44ch">' + msg + '</p></div></div>');
  }

  fetchJSON('manifest.json').then(function (man) {
    state.manifest = man;
    state.passes = man.passes || [];
    $('sample-chip').hidden = !man.sample;
    $('delay-chip').textContent = man.rules.delay_hours + ' h publication delay';
    $('attrib-rules').textContent = 'delayed ' + man.rules.delay_hours + ' h · ' +
      (man.rules.masked ? 'military areas masked' : 'no area mask configured');
    $('floor-note').textContent = 'Individual positions are shown at or above ' +
      man.rules.size_floor_m + ' m. Anything smaller is counted per ' +
      man.rules.aggregate_cell_km + ' km cell, never placed.';
    document.title = man.title + (man.sample ? ' (sample data)' : '');

    return buildMap(man.bbox).then(function () {
      addLayers(man);
      buildLayerRows(man);
      wire();
      updateScale();
      if (!state.passes.length) { fail('manifest has no passes yet'); return; }
      buildTrack();
      loadPass(0);
    });
  }).catch(function (e) {
    console.error(e);
    fail('could not load manifest');
  });
})();
