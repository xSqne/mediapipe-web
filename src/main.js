import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let handLandmarker;

// Hand landmark connections for drawing skeleton
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],         // Index finger
    [0, 9], [9, 10], [10, 11], [11, 12],    // Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20],  // Pinky
    [5, 9], [9, 13], [13, 17]               // Palm connections
];

// Initialize MediaPipe HandLandmarker
async function initializeHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
}

function detectHands() {
    // Check handLandmarker and video dimensions are valid/initialized
    if (handLandmarker && video.videoWidth > 0 && video.videoHeight > 0) {
        const results = handLandmarker.detectForVideo(video, performance.now());
        
        // Set canvas size to match the video size 
        const videoRect = video.getBoundingClientRect();
        canvas.width = videoRect.width;
        canvas.height = videoRect.height;
        canvas.style.width = videoRect.width + 'px';
        canvas.style.height = videoRect.height + 'px';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // If hands are detected:
        if (results.landmarks && results.landmarks.length > 0) {
            // Loop through each detected hand
            for (let i = 0; i < results.landmarks.length; i++) {
                const landmarks = results.landmarks[i];
                
                // Draw hand landmarks and connections
                drawHandLandmarks(landmarks);
            }
        }
    }
    
    // Continue detection loop
    requestAnimationFrame(detectHands);
}

function drawHandLandmarks(landmarks) {
    ctx.lineWidth = 2;
    
    // Get wrist Z position
    const wristZ = landmarks[0].z;
    
    // Calculate relative Z depths from wrist for all landmarks
    const relativeZs = landmarks.map(landmark => landmark.z - wristZ);
    
    // Use fixed Z range to reduce sensitivity
    const zRange = 0.2;
    
    // Draw connections
    HAND_CONNECTIONS.forEach(([start, end]) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];
        
        // Calculate average relative Z for the connection line
        const avgRelativeZ = (relativeZs[start] + relativeZs[end]) / 2;
        
        // Normalize Z with fixed range: clamp between -0.1 and +0.1 relative to wrist
        const clampedZ = Math.max(-0.1, Math.min(0.1, avgRelativeZ));
        const t = (clampedZ + 0.1) / 0.2;
        
        // Color gradient: farther from wrist = red, same as wrist = blue, closer = green
        const red = Math.round(255 * t);         // Red when farther from wrist (folded back)
        const green = Math.round(255 * (1 - t)); // Green when closer to wrist
        const blue = Math.round(255 * (1 - Math.abs(t - 0.5) * 2)); // Blue at wrist level
        
        ctx.strokeStyle = `rgb(${red}, ${green}, ${blue})`;
        
        // Flip X coordinates to match mirrored video
        const startX = canvas.width - (startPoint.x * canvas.width);
        const startY = startPoint.y * canvas.height;
        const endX = canvas.width - (endPoint.x * canvas.width);
        const endY = endPoint.y * canvas.height;
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    });
    
    // Draw landmark points (same logic as connections)
    landmarks.forEach((landmark, index) => {
        // Flip X coordinate to match mirrored video
        const x = canvas.width - (landmark.x * canvas.width);
        const y = landmark.y * canvas.height;
        
        // Calculate relative Z depth from wrist
        const relativeZ = relativeZs[index];
        
        const clampedZ = Math.max(-0.1, Math.min(0.1, relativeZ));
        const t = (clampedZ + 0.1) / 0.2;
        
        const red = Math.round(255 * t);         
        const green = Math.round(255 * (1 - t)); 
        const blue = Math.round(255 * (1 - Math.abs(t - 0.5) * 2)); 
        
        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        
        ctx.beginPath();
        ctx.arc(x, y, index === 0 ? 8 : 4, 0, 2 * Math.PI); // Larger circle for wrist
        ctx.fill();
    });
}

// Setup webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(async (stream) => {
    video.srcObject = stream;
    
    // Start hand detection once video is loaded
    video.addEventListener('loadeddata', async () => {
        await initializeHandLandmarker();
        console.log("HandLandmarker initialized");
        detectHands();
    });
})
.catch(err => {
    video.style.display = 'none';
    alert('Could not access webcam: ' + err.message);
});