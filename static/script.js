document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const scanningAnimation = document.getElementById('scanningAnimation');
    const meterFill = document.getElementById('meterFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const resultBadge = document.getElementById('resultBadge');
    const videoPreviewCard = document.getElementById('videoPreviewCard');
    const previewVideo = document.getElementById('previewVideo');
    const downloadWatermarked = document.getElementById('downloadWatermarked');

    let currentVideoFile = null;
    let watermarkedPath = null;

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        currentVideoFile = formData.get('video');
        
        // Show scanning animation
        scanningAnimation.style.display = 'flex';
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Update UI with results
            updateResults(data);
            
            // If video is fake, show the preview
            if (data.final_decision === 'FAKE') {
                showVideoPreview(currentVideoFile);
                watermarkedPath = data.watermarked_path;
            } else {
                hideVideoPreview();
                watermarkedPath = null;
            }
            
        } catch (error) {
            resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            hideVideoPreview();
            watermarkedPath = null;
        } finally {
            // Hide scanning animation
            scanningAnimation.style.display = 'none';
        }
    });

    function showVideoPreview(videoFile) {
        videoPreviewCard.style.display = 'block';
        const videoURL = URL.createObjectURL(videoFile);
        previewVideo.src = videoURL;
    }

    function hideVideoPreview() {
        videoPreviewCard.style.display = 'none';
        if (previewVideo.src) {
            URL.revokeObjectURL(previewVideo.src);
            previewVideo.src = '';
        }
    }

    downloadWatermarked.addEventListener('click', function() {
        if (watermarkedPath) {
            // Download the watermarked video from the server
            window.location.href = `/download/${watermarkedPath}`;
        }
    });

    function updateResults(data) {
        // Update confidence meter
        const confidence = data.final_decision === 'FAKE' ? 100 : 0;
        meterFill.style.width = `${confidence}%`;
        confidenceValue.textContent = `${confidence}%`;
        
        // Update result badge
        resultBadge.textContent = data.final_decision;
        resultBadge.className = `badge ${data.final_decision.toLowerCase()}`;
        
        // Update timestamp
        const timestamp = document.getElementById('resultTimestamp');
        timestamp.textContent = new Date().toLocaleString();
        
        // Update visual details
        const visualDetails = document.getElementById('visualDetails');
        visualDetails.innerHTML = `
            <li>Total Frames Analyzed: ${data.total_fake_frames + data.total_real_frames}</li>
            <li>Fake Frames: ${data.total_fake_frames}</li>
            <li>Real Frames: ${data.total_real_frames}</li>
        `;
        
        // Update stats
        document.getElementById('genuineCount').textContent = data.total_real_frames;
        document.getElementById('fakeCount').textContent = data.total_fake_frames;
    }
}); 