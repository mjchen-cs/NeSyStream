// Runs dash.js sessions in headless Chromium against the trace-shaped segment server.
// node run_sessions.js out_dir method1,method2 trace1 [trace2 ...]
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs'), path = require('path');
const [outDir, methods, ...traces] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  let browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
  let port = 9100;
  for (const tr of traces) {
    for (const m of methods.split(',')) {
      const tag = `${path.basename(path.dirname(tr))}__${path.basename(tr)}__${m}`;
      const out = path.join(outDir, tag + '.json');
      if (fs.existsSync(out)) continue;
      const segLog = path.join(outDir, tag + '.seglog.json');
      port += 1;
      const srv = spawn('python3', [path.join(__dirname, 'seg_server.py'), String(port), tr, segLog], { stdio: 'ignore' });
      await sleep(800);
      try {
      if (!browser.isConnected()) browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
      const page = await browser.newPage();
      const t0 = Date.now();
      await page.goto(`http://127.0.0.1:${port}/index.html?method=${m}&sid=${encodeURIComponent(tag)}&abr=http://127.0.0.1:8333`);
      let log = null;
      while (Date.now() - t0 < 150000) {
        await sleep(1000);
        log = await page.evaluate(() => window.__log);
        if (log && log.done) break;
      }
      const q = await page.evaluate(() => { const v = document.querySelector('video'); const p = v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality() : {}; return { dropped: p.droppedVideoFrames, total: p.totalVideoFrames, ct: v.currentTime }; });
      await page.close(); srv.kill();
      const seg = fs.existsSync(segLog) ? JSON.parse(fs.readFileSync(segLog)) : [];
      fs.writeFileSync(out, JSON.stringify({ tag, method: m, trace: tr, wall_s: (Date.now() - t0) / 1000, quality: q, log, seg }));
      console.log(tag, 'done', log && log.done, 'segs', seg.length, 'wall', ((Date.now() - t0) / 1000).toFixed(0), 'dropped', q.dropped, '/', q.total);
      } catch (err) {
        console.log(tag, 'FAILED', String(err).slice(0, 120)); srv.kill();
        try { await browser.close(); } catch (e) {}
        browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
      }
    }
  }
  await browser.close();
})();
