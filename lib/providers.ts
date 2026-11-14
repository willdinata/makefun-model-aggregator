import { readFileSync } from 'node:fs';

export const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

const schemas = {
  providers: 'free-model-aggregator-01/providers/v1',
  ratings: 'free-model-aggregator-01/ratings/v1',
  workload: 'free-model-aggregator-01/workload/v1',
};

function objectEnd(text, start) {
  let depth = 0;
  let quoted = false;
  for (let index = start; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '\\') index += 1;
      else if (character === '"') quoted = false;
    } else if (character === '"') quoted = true;
    else if (character === '{') depth += 1;
    else if (character === '}') {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function documents(text) {
  const found = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end < 0) continue;
    try {
      const value = JSON.parse(text.slice(start, end + 1));
      if (isObject(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // The brief contains prose and JSON; only complete schema documents are useful.
    }
  }
  return found;
}

function text(value) {
  return typeof value === 'string' ? value : '';
}

function numeric(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function parseProvider(row) {
  if (!isObject(row)) return null;
  const id = text(row.id);
  const wire = ['openai', 'anthropic', 'gemini'].includes(row.wire) ? row.wire : null;
  if (!id || !wire) return null;
  const auth = ['bearer', 'x-api-key', 'x-goog-api-key', 'none'].includes(row.auth) ? row.auth : 'none';
  return {
    id,
    wire,
    auth,
    key_env: text(row.key_env),
    key: text(row.key),
    path: text(row.path) || text(row.base_path) || text(row.base_url) || text(row.base),
  };
}

function parseRating(row) {
  if (!isObject(row) || !text(row.model) || !text(row.provider)) return null;
  return {
    model: row.model,
    provider: row.provider,
    rating_milli: numeric(row.rating_milli) ?? 0,
    context_length: numeric(row.context_length),
    tools: row.tools === true,
    vision: row.vision === true,
    json: row.json === true,
    free: row.free === true,
  };
}

function parseRequirement(value) {
  if (!isObject(value)) return {};
  const requirement = {
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
  };
  if (typeof value.min_context === 'number' && Number.isFinite(value.min_context)) {
    requirement.min_context = value.min_context;
  }
  return requirement;
}

function parseWorkload(row) {
  if (!isObject(row) || !text(row.id)) return null;
  return {
    id: row.id,
    api: row.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: row.stream === true,
    weight: row.weight === 2 ? 2 : 1,
    requires: parseRequirement(row.requires),
  };
}

export function readBrief(path) {
  const empty = { providers: [], ratings: [], manifest: [] };
  if (!path) return empty;
  try {
    const docs = documents(readFileSync(path, 'utf8'));
    const providerDoc = docs.find((doc) => doc.schema === schemas.providers && Array.isArray(doc.providers));
    const ratingDoc = docs.find((doc) => doc.schema === schemas.ratings && Array.isArray(doc.ratings));
    const workloadDoc = docs.find((doc) => doc.schema === schemas.workload && Array.isArray(doc.manifest));
    return {
      providers: (providerDoc?.providers ?? []).map(parseProvider).filter(Boolean),
      ratings: (ratingDoc?.ratings ?? []).map(parseRating).filter(Boolean),
      manifest: (workloadDoc?.manifest ?? []).map(parseWorkload).filter(Boolean),
    };
  } catch (error) {
    process.stderr.write(`aggregator: cannot read brief: ${error.message}\n`);
    return empty;
  }
}

export function ratingFor(brief, provider, model) {
  return brief.ratings.find((row) => row.provider === provider && row.model === model) ?? null;
}

export function providerUrl(envUrl, provider) {
  const base = provider.path.trim().replace(/\/+$/, '');
  if (/^https?:\/\//i.test(base)) return base;
  return `${envUrl.replace(/\/+$/, '')}${base.startsWith('/') ? '' : '/'}${base}`;
}

export function authHeaders(provider) {
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

function freeFromPrice(value) {
  if (typeof value.free === 'boolean') return value.free;
  if (!isObject(value.pricing)) return null;
  const prices = Object.values(value.pricing).map((price) => Number(price));
  return prices.length > 0 && prices.every((price) => Number.isFinite(price) && price === 0);
}

function listedModels(wire, body) {
  if (!isObject(body)) return [];
  const rows = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((row) => {
    if (!isObject(row)) return [];
    const id = wire === 'gemini' ? text(row.name).replace(/^models\//, '') : text(row.id);
    if (!id) return [];
    return [{
      id,
      free: freeFromPrice(row),
      context_length: numeric(row.context_length) ?? numeric(row.context_window) ?? numeric(row.inputTokenLimit),
    }];
  });
}

export async function probe(envUrl, provider) {
  const endpoint = provider.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
  const headers = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const response = await fetch(`${providerUrl(envUrl, provider)}${endpoint}`, {
      headers,
      signal: AbortSignal.timeout(3500),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return { provider, ok: true, models: listedModels(provider.wire, await response.json()) };
  } catch (error) {
    process.stderr.write(`aggregator: catalog probe ${provider.id} failed: ${error.message}\n`);
    return { provider, ok: false, models: [] };
  }
}

function modelRowsFor(probeResult, brief) {
  if (!probeResult.ok) {
    return brief.ratings
      .filter((rating) => rating.provider === probeResult.provider.id)
      .map((rating) => ({ id: rating.model, free: rating.free, context_length: rating.context_length }));
  }
  const unique = new Map();
  for (const model of probeResult.models) {
    const rating = ratingFor(brief, probeResult.provider.id, model.id);
    unique.set(model.id, {
      id: model.id,
      free: rating ? rating.free : model.free ?? false,
      context_length: rating ? rating.context_length : model.context_length,
    });
  }
  return [...unique.values()];
}

function compareIds(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

export function catalogFrom(probes, brief) {
  return {
    providers: probes.map(({ provider, ...result }) => ({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.key_env,
      wire: provider.wire,
      models: modelRowsFor({ provider, ...result }, brief).sort((a, b) => compareIds(a.id, b.id)),
    })),
  };
}

export function canServe(requirement, rating) {
  if (requirement.tools && !rating.tools) return false;
  if (requirement.vision && !rating.vision) return false;
  if (requirement.json && !rating.json) return false;
  if (requirement.min_context !== undefined
      && (rating.context_length === null || rating.context_length < requirement.min_context)) return false;
  return true;
}

export function rankedCandidates(brief, probes, catalog, requirement, deadProviders) {
  const probeById = new Map(probes.map((item) => [item.provider.id, item]));
  const catalogById = new Map(catalog.providers.map((provider) => [provider.id, provider]));
  const candidates = [];
  for (const provider of brief.providers) {
    const probeResult = probeById.get(provider.id);
    const catalogProvider = catalogById.get(provider.id);
    if (!probeResult?.ok || !catalogProvider) continue;
    for (const model of catalogProvider.models) {
      const rating = ratingFor(brief, provider.id, model.id);
      const listed = probeResult.models.find((item) => item.id === model.id);
      if (!rating || !listed || rating.free !== true || model.free !== true || listed.free !== true) continue;
      if (!canServe(requirement, rating)) continue;
      candidates.push({ provider, model, rating });
    }
  }
  candidates.sort((a, b) => b.rating.rating_milli - a.rating.rating_milli
    || compareIds(a.provider.id, b.provider.id)
    || compareIds(a.model.id, b.model.id));
  return candidates.filter((candidate) => !deadProviders.has(candidate.provider.id));
}
