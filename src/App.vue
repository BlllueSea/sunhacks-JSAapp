<template>
  <v-app>
    <v-main>
      <div class="user-container">
        <div id="canvas-wrapper">
          <canvas id="output" style=""></canvas>
          <video id="video" playsinline></video>
        </div>
        <div id="scatter-gl-container"></div>
      </div>
    </v-main>
  </v-app>
</template>

<script>
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';
import * as handpose from '@tensorflow-models/handpose';
import ScatterGL from 'scatter-gl';
import dat from 'dat.gui';
import Stats from 'stats.js';
// tfjsWasm.setWasmPath(
//     `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@$
//         tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`);


let videoWidth, videoHeight, rafID, ctx, canvas, ANCHOR_POINTS,
    scatterGLHasInitialized = false, scatterGL, fingerLookupIndices = {
      thumb: [0, 1, 2, 3, 4],
      indexFinger: [0, 5, 6, 7, 8],
      middleFinger: [0, 9, 10, 11, 12],
      ringFinger: [0, 13, 14, 15, 16],
      pinky: [0, 17, 18, 19, 20]
    };  // for rendering each finger as a polyline

let x_obj = 100, y_obj = 100;
let x_goal = 550, y_goal = 400;
let thumbs, pinky, middle;
let dist_t_p, dist_h_obj, dist_goal_obj;
let x1, y1, x2, y2, x3, y3;

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;

// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = true;

const state = {
  backend: 'webgl'
};

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

if (renderPointcloud) {
  state.renderPointcloud = true;
}

function setupDatGui(video) {
  const gui = new dat.GUI();
  gui.add(state, 'backend', ['webgl'])
      .onChange(async backend => {
        window.cancelAnimationFrame(rafID);
        await tf.setBackend(backend);
        landmarksRealTime(video);
      });

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
          render ? 'inline-block' : 'none';
    });
  }
}

function drawPoint(y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawPath(points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

// Muraishi: Draw hands path
function drawKeypoints(keypoints) {
  const keypointsArray = keypoints;
  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(x , y , 3);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
    drawPath(points, false);
  }
}

function drawObject(y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

let model;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width:  VIDEO_WIDTH,
      height: VIDEO_HEIGHT
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

async function main() {
  await tf.setBackend(state.backend);
  model = await handpose.load();
  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  setupDatGui(video);

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = 'red';
  ctx.fillStyle = 'red';

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // These anchor points allow the hand pointcloud to resize according to its
  // position in the input.
  ANCHOR_POINTS = [
    [0, 0, 0], [0, -VIDEO_HEIGHT, 0], [-VIDEO_WIDTH, 0, 0],
    [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
  ];

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
        `width: ${VIDEO_WIDTH}px; height: ${VIDEO_HEIGHT}px;`;

    // scatterGL = new ScatterGL(
    //     document.querySelector('#scatter-gl-container'),
    //     {'rotateOnStart': false, 'selectEnabled': false});
  }
  landmarksRealTime(video);
  // objectRealTime();
}

const landmarksRealTime = async (video) => {
  async function frameLandmarks() {
    stats.begin();
    ctx.drawImage(
        video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
        canvas.height);

    ctx.beginPath();
    ctx.arc(x_goal, y_goal, 50, 0, 2 * Math.PI);
    ctx.stroke();

    // muraishi: get predictions from the model
    const predictions = await model.estimateHands(video);
    if (predictions.length > 0) {
      const result = predictions[0].landmarks;
      const result_details = predictions[0].annotations;

      thumbs = result_details.thumb;
      pinky = result_details.pinky;
      middle = result_details.middleFinger;
      console.log(predictions[0]);
      y1=thumbs[0][0];
      y2=pinky[0][0];
      y3=middle[3][0];
      x1=thumbs[0][1];
      x2=pinky[0][1];
      x3=middle[3][1];

      dist_t_p = Math.sqrt(Math.pow(x1-x2,2) + Math.pow(y1-y2,2));
      dist_h_obj = Math.sqrt(Math.pow(x3-x_obj,2) + Math.pow(y3-y_obj,2));

      if (dist_t_p < 100 && dist_h_obj < 60) {
        x_obj = x3;
        y_obj = y3;
      }
      dist_goal_obj = Math.sqrt(Math.pow(x_goal-x_obj,2) + Math.pow(y_goal-y_obj,2));
      if (dist_goal_obj < 100) {
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.strokeStyle = 'blue';
        ctx.fillStyle = 'blue';
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      drawKeypoints(result, result_details);
      drawObject(x_obj, y_obj, 30);


      if (renderPointcloud === true && scatterGL != null) {
        const pointsData = result.map(point => {
          return [-point[0], -point[1], -point[2]];
        });

        const dataset =
            new ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

        if (!scatterGLHasInitialized) {
          scatterGL.render(dataset);

          const fingers = Object.keys(fingerLookupIndices);

          scatterGL.setSequences(
              fingers.map(finger => ({indices: fingerLookupIndices[finger]})));
          scatterGL.setPointColorer((index) => {
            if (index < pointsData.length) {
              return 'steelblue';
            }
            return 'white';  // Hide.
          });
        } else {
          scatterGL.updateDataset(dataset);
        }
        scatterGLHasInitialized = true;
      }
    }
    stats.end();
    rafID = requestAnimationFrame(frameLandmarks);
  }

  frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

export default {
  name: 'App',

  components: {
    // HelloWorld,
  },
  data: function() {
    return {
    };
  },
  methods: {
  },
  created: function() {
    main();
  },
  mounted: function() {
  }
};
</script>
<style lang="scss">
#video {
  -webkit-transform: scaleX(-1);
  transform: scaleX(-1);
  visibility: hidden;
  width: auto;
  height: auto;
  position: absolute;
}
</style>
