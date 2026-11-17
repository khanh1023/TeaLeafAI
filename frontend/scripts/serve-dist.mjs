import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { spawn } from 'node:child_process';
const root = fileURLToPath(new URL('../dist/', import.meta.url));
const port = Number(process.env.PORT || 5173);
const mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.svg':'image/svg+xml','.wasm':'application/wasm','.png':'image/png','.onnx':'application/octet-stream'};
const server = http.createServer(async (req,res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    let file = path.resolve(root, '.' + pathname);
    if (file !== root && !file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    let info = await stat(file).catch(() => null);
    if (info?.isDirectory()) { file = path.join(file,'index.html'); info = await stat(file).catch(() => null); }
    if (!info?.isFile()) {
      if (path.extname(pathname)) { res.writeHead(404).end('Not found'); return; }
      file = path.join(root,'index.html');
    }
    const data = await readFile(file);
    res.writeHead(200, {'Content-Type':mime[path.extname(file)] || 'application/octet-stream','X-Content-Type-Options':'nosniff'});
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch { res.writeHead(500).end('Cannot read website. Run npm run build inside frontend first.'); }
});
server.on('error',error => { console.error(`Cannot start TeaLeafAI: ${error.message}. Close the old server or use another PORT.`);process.exit(1); });
server.listen(port,'127.0.0.1',() => {
  const url = `http://127.0.0.1:${port}`;
  console.log(`TeaLeafAI is ready: ${url}\nKeep this window open. Ctrl+C to stop. No Python or API key needed.`);
  if (process.argv.includes('--open')) {
    const command = process.platform === 'win32' ? ['cmd',['/c','start','',url]] : process.platform === 'darwin' ? ['open',[url]] : ['xdg-open',[url]];
    const child = spawn(command[0],command[1],{stdio:'ignore'}); child.on('error',()=>{}); child.unref();
  }
});
