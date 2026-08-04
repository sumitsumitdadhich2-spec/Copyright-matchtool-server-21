import fs from 'fs';

async function processVideo(filePath) {
  const formData = new FormData();
  formData.append('video', new Blob([fs.readFileSync(filePath)]), filePath);
  
  const uploadRes = await fetch('http://localhost:3001/api/upload', {
    method: 'POST',
    body: formData,
  });
  
  const { jobId } = await uploadRes.json();
  console.log(`Uploaded ${filePath}, Job ID: ${jobId}`);
  
  while (true) {
    const statusRes = await fetch(`http://localhost:3001/api/status/${jobId}`);
    const status = await statusRes.json();
    console.log(`Job ${jobId} status: ${status.status}, processed: ${status.processedFrames}/${status.totalFrames}`);
    if (status.status === 'completed') {
      break;
    } else if (status.status === 'failed') {
      throw new Error(`Processing failed: ${status.error}`);
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  const resultRes = await fetch(`http://localhost:3001/api/result/${jobId}`);
  const results = await resultRes.json();
  console.log(`Job ${jobId} finished. Fetched ${results.length} fingerprints.`);
  return results;
}

async function main() {
  console.log("Processing reference video...");
  const refFps = await processVideo('real_video.mp4');
  console.log("Reference processing done.");
  
  // Create a 10s clip for target (already have test_10s.mp4? Let's check or use test_2m.mp4 again but just 10s)
}
main().catch(console.error);
