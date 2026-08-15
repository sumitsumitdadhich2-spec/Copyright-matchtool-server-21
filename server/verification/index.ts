/**
 * Public surface of the verification/candidate system.
 * The rest of the app should import from here and nowhere deeper.
 */

export {
  verifyMatchedSegments,
  recheckSegment,
  type VerifyRequest,
  type VerifyResult,
  type VerifySummary,
  type RecheckRequest,
  type RecheckResult,
} from './verify';

export {
  readRecord,
  readAllRecords,
  listRecordIndexes,
  deleteRecordsForJob,
  writeRecord,
  type CandidateRecord,
  type CandidateVerdict,
  type VerificationRecord,
} from './store';

export { flagTimelineOutliers } from './timeline';
