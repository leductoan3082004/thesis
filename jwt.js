const crypto = require('crypto');
const fs = require('fs');

const args = process.argv.slice(2);
const cli = {};
for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--sub') {
        cli.sub = args[++i];
    } else if (arg === '--role') {
        cli.role = args[++i];
    } else if (arg === '--state') {
        cli.state = args[++i];
    } else if (arg === '--ttl-days') {
        cli.ttlDays = parseInt(args[++i], 10);
    } else {
        throw new Error(`Unknown flag ${arg}`);
    }
}

const ttlDays = Number.isFinite(cli.ttlDays) ? cli.ttlDays : 30;
const payload = {
    sub: cli.sub || process.env.JWT_SUB || 'trainer-node-001',
    state: cli.state || process.env.JWT_STATE || 'system',
    role: cli.role || process.env.JWT_ROLE || 'trainer',
    exp: Math.floor(Date.now() / 1000) + 3600 * 24 * ttlDays,
};

const alg = (process.env.JWT_ALG || 'HS256').toUpperCase();
const header = { alg, typ: 'JWT' };

const base64url = (value) =>
    Buffer.from(JSON.stringify(value)).toString('base64url');

const unsigned = `${base64url(header)}.${base64url(payload)}`;
let signature;

if (alg === 'HS256') {
    const secret = process.env.AUTH_JWT_SECRET || 'replace-me';
    signature = crypto
        .createHmac('sha256', secret)
        .update(unsigned)
        .digest('base64url');
} else if (alg === 'EDDSA') {
    const keyPath = process.env.TRAINER_PRIVATE_KEY || 'trainer_ed25519_sk.pem';
    const key = fs.readFileSync(keyPath);
    signature = crypto
        .sign(null, Buffer.from(unsigned), key)
        .toString('base64url');
} else {
    throw new Error(`Unsupported alg ${alg}. Use HS256 or EdDSA.`);
}

console.log(`${unsigned}.${signature}`);
