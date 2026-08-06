/**
 * Sanity test for server/candidate-embedding-rank.ts (crop-robust ranking).
 *
 * A 9:16 RIGHT-crop short taken from t=10s of a synthetic long video must
 * rank the correct candidate (movieStart=10) ABOVE a wrong one
 * (movieStart=25), even though the short shows only the right slice of the
 * widescreen frame. Run: npx tsx test_crop_rank.mts
 * (needs ffmpeg on PATH, or FFMPEG_BIN pointing at a binary)
 */
import { spawnSync } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import { rankCandidatesCropRobust } from './server/candidate-embedding-rank';

const FF = process.env.FFMPEG_BIN || 'ffmpeg';
const LONG = path.join(os.tmpdir(), 'crop_rank_long.mp4');
const SHORT = path.join(os.tmpdir(), 'crop_rank_short.mp4');

function run(args: string[]) {
  const r = spawnSync(FF, ['-v', 'error', '-y', ...args], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${r.stderr}`);
}

// Long: 30s of mandelbrot (visually rich, position-dependent content).
run(['-f', 'lavfi', '-i', 'mandelbrot=size=1280x720:rate=10', '-t', '30', '-pix_fmt', 'yuv420p', LONG]);
// Short: seconds 10-13 of the long video, RIGHT 9:16 crop.
run(['-ss', '10', '-i', LONG, '-t', '3', '-vf', 'crop=w=min(iw\\,ih*9/16):h=ih:x=iw-ow:y=0', '-pix_fmt', 'yuv420p', SHORT]);

function seg(movieStart: number) {
  return {
    segment: {
      shortStart: 0, shortEnd: 3,
      movieStart, movieEnd: movieStart + 3,
      confidence: 90,
      matchSequence: [
        { shortTime: 0.5, movieTime: movieStart + 0.5, similarity: 90 },
        { shortTime: 1.5, movieTime: movieStart + 1.5, similarity: 90 },
        { shortTime: 2.5, movieTime: movieStart + 2.5, similarity: 90 },
      ],
    } as any,
  };
}

// Candidate 0 = WRONG (t=25), candidate 1 = CORRECT (t=10). Original order
// would try the wrong one first — ranking must flip it.
const candidates = [seg(25), seg(10)];

const ranked = await rankCandidatesCropRobust(candidates, [0, 1], SHORT, LONG, 'Test');
console.log('[v0] ranked order:', ranked);
if (!ranked) {
  console.log('[v0] RESULT: ranking unavailable (model load failed?)');
  process.exit(1);
}
console.log(ranked[0] === 1 ? '[v0] RESULT: PASS — correct candidate ranked first' : '[v0] RESULT: FAIL — wrong order');
process.exit(ranked[0] === 1 ? 0 : 1);
