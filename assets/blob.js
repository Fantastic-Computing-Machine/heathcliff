/* =================================================================
   Heathcliff Blob UI – blob.js
   Three.js scene: organic flowing blob with multi-octave GPU noise,
   localized touch ripples with spring-bounce physics,
   intensity control, and state engine.
   ================================================================= */

/* global THREE */

// =====================================================================
//  INTENSITY CONTROL  (0 → 1)
// =====================================================================
let intensity = 0.5;

(function readIntensityParam() {
  const params = new URLSearchParams(window.location.search);
  const v = parseFloat(params.get("intensity"));
  if (!isNaN(v)) intensity = Math.max(0, Math.min(1, v));
})();

function setBlobIntensity(v) {
  intensity = Math.max(0, Math.min(1, v));
}
window.setBlobIntensity = setBlobIntensity;

// =====================================================================
//  TOUCH / CLICK RIPPLE SYSTEM
//  Up to 5 concurrent localized ripples that propagate from the hit point.
//  Plus a global spring-bounce for the whole body.
// =====================================================================
const MAX_RIPPLES = 5;
const ripples = []; // { origin: THREE.Vector3, time: float, strength: float }

let bounceImpulse = 0;
let bounceVelocity = 0;
const BOUNCE_SPRING = -12.0;
const BOUNCE_DAMPING = 0.92;

// =====================================================================
//  STATE DEFINITIONS
//  High noise amplitudes = visibly non-spherical, organic, blobby.
//  Three octaves of noise give flowing, liquid deformation.
// =====================================================================
const STATES = {
  idle: {
    speed:       0.35,
    noiseAmp:    0.35,
    noiseAmp2:   0.15,
    noiseFreq:   0.8,
    noiseFreq2:  1.6,
    colorA: new THREE.Color(0xc4b0ff),
    colorB: new THREE.Color(0xffb0c8),
    emissive: new THREE.Color(0x12081e),
    rotSpeed:    0.15,
    pulseAmp:    0.04,
    pulseFreq:   0.6,
    label: "Heathcliff",
  },
  listening: {
    speed:       0.55,
    noiseAmp:    0.40,
    noiseAmp2:   0.18,
    noiseFreq:   0.9,
    noiseFreq2:  1.8,
    colorA: new THREE.Color(0xd4a0ff),
    colorB: new THREE.Color(0x90c0ff),
    emissive: new THREE.Color(0x0e0818),
    rotSpeed:    0.22,
    pulseAmp:    0.06,
    pulseFreq:   0.9,
    label: "Listening...",
  },
  thinking: {
    speed:       0.70,
    noiseAmp:    0.45,
    noiseAmp2:   0.22,
    noiseFreq:   1.0,
    noiseFreq2:  2.0,
    colorA: new THREE.Color(0xd9a0ff),
    colorB: new THREE.Color(0xffa0b8),
    emissive: new THREE.Color(0x140a20),
    rotSpeed:    0.30,
    pulseAmp:    0.07,
    pulseFreq:   1.2,
    label: "Thinking...",
  },
  speaking: {
    speed:       0.50,
    noiseAmp:    0.38,
    noiseAmp2:   0.16,
    noiseFreq:   0.85,
    noiseFreq2:  1.7,
    colorA: new THREE.Color(0xb8d0ff),
    colorB: new THREE.Color(0xffd0a0),
    emissive: new THREE.Color(0x0c0a18),
    rotSpeed:    0.25,
    pulseAmp:    0.05,
    pulseFreq:   1.0,
    label: "Speaking...",
  },
};

// =====================================================================
//  STATE ENGINE
// =====================================================================
let currentStateName = "idle";
let targetState = { ...STATES.idle };

const lerpState = {
  speed:      STATES.idle.speed,
  noiseAmp:   STATES.idle.noiseAmp,
  noiseAmp2:  STATES.idle.noiseAmp2,
  noiseFreq:  STATES.idle.noiseFreq,
  noiseFreq2: STATES.idle.noiseFreq2,
  colorA:     STATES.idle.colorA.clone(),
  colorB:     STATES.idle.colorB.clone(),
  emissive:   STATES.idle.emissive.clone(),
  rotSpeed:   STATES.idle.rotSpeed,
  pulseAmp:   STATES.idle.pulseAmp,
  pulseFreq:  STATES.idle.pulseFreq,
};

function setState(name) {
  if (!STATES[name]) return;
  currentStateName = name;
  targetState = STATES[name];

  const label = document.getElementById("status-label");
  if (label) {
    label.textContent = targetState.label;
    label.classList.toggle("active", name !== "idle");
  }
}

window.setBlobState = setState;

// =====================================================================
//  THREE.JS SCENE
// =====================================================================
const canvas = document.getElementById("blob-canvas");

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.1, 100
);
camera.position.set(0, 0, 4.2);

// ── Sphere Geometry ─────────────────────────────────────────────────
const geo = new THREE.SphereGeometry(1, 256, 128);

// ── Vertex Shader: 3-octave simplex noise + localized ripple waves ──
const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uNoiseAmp;
  uniform float uNoiseAmp2;
  uniform float uNoiseFreq;
  uniform float uNoiseFreq2;
  uniform float uPulse;
  uniform float uBounce;

  // Ripple uniforms (up to 5)
  uniform vec3  uRippleOrigins[5];
  uniform float uRippleTimes[5];
  uniform float uRippleStrengths[5];
  uniform int   uRippleCount;

  varying vec3  vNormal;
  varying vec3  vWorldPos;
  varying float vDisplacement;

  // ── Simplex 3D noise (Ashima Arts / Stefan Gustavson) ──────────
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  void main() {
    vec3 pos = position;
    vec3 nrm = normal;
    float t = uTime;

    // ── Octave 1: large blobby lobes ──────────────────────────────
    float n1 = snoise(pos * uNoiseFreq + t * 0.6);

    // ── Octave 2: medium flowing detail ───────────────────────────
    float n2 = snoise(pos * uNoiseFreq2 + t * 0.4 + 10.0);

    // ── Octave 3: subtle fine-grain wobble ────────────────────────
    float n3 = snoise(pos * uNoiseFreq2 * 2.0 + t * 0.8 + 20.0) * 0.3;

    float noise = n1 * uNoiseAmp + n2 * uNoiseAmp2 + n3 * uNoiseAmp2 * 0.5;

    // ── Asymmetric base deformation (permanently breaks spherical shape)
    float asymmetry = snoise(pos * 0.5 + 5.0) * 0.12;

    // ── Pulse (breathing) + bounce ────────────────────────────────
    float disp = noise + asymmetry + uPulse + uBounce;

    // ── Localized ripple waves from touch points ──────────────────
    for (int i = 0; i < 5; i++) {
      if (i >= uRippleCount) break;
      float dist = distance(pos, uRippleOrigins[i]);
      float age = uRippleTimes[i];
      float waveFront = age * 2.5;
      float ringDist = abs(dist - waveFront);
      float ripple = sin(ringDist * 12.0 - age * 8.0)
                     * exp(-ringDist * 3.0)
                     * exp(-age * 2.0)
                     * uRippleStrengths[i];
      disp += ripple;
    }

    vec3 displaced = pos + nrm * disp;

    vDisplacement = disp;
    vWorldPos     = (modelMatrix * vec4(displaced, 1.0)).xyz;
    vNormal       = normalize(normalMatrix * nrm);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3  uColorA;
  uniform vec3  uColorB;
  uniform vec3  uEmissive;
  uniform float uTime;

  varying vec3  vNormal;
  varying vec3  vWorldPos;
  varying float vDisplacement;

  void main() {
    vec3 viewDir = normalize(cameraPosition - vWorldPos);

    // Fresnel for soft edge glow
    float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0);

    // Color gradient driven by displacement
    float mix_ = vDisplacement * 2.0 + 0.5;
    mix_ += sin(vWorldPos.y * 2.5 + uTime * 0.3) * 0.2;
    mix_ = clamp(mix_, 0.0, 1.0);

    vec3 col = mix(uColorA, uColorB, mix_);

    // Fresnel glow
    vec3 fresnelCol = mix(uColorA, vec3(1.0), 0.4);
    col += fresnelCol * fresnel * 0.35;

    // Emissive base
    col += uEmissive;

    // Soft diffuse lighting
    vec3 lightDir = normalize(vec3(0.8, 1.0, 0.6));
    float diff = max(dot(vNormal, lightDir), 0.0) * 0.4 + 0.6;
    col *= diff;

    // Specular highlight
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(vNormal, halfDir), 0.0), 60.0);
    col += vec3(1.0) * spec * 0.15;

    gl_FragColor = vec4(col, 0.94);
  }
`;

// ── Ripple uniform arrays ───────────────────────────────────────────
const rippleOriginsArr = [];
const rippleTimesArr = [];
const rippleStrengthsArr = [];
for (let i = 0; i < MAX_RIPPLES; i++) {
  rippleOriginsArr.push(new THREE.Vector3(0, 0, 0));
  rippleTimesArr.push(0);
  rippleStrengthsArr.push(0);
}

const mat = new THREE.ShaderMaterial({
  vertexShader,
  fragmentShader,
  uniforms: {
    uColorA:          { value: lerpState.colorA },
    uColorB:          { value: lerpState.colorB },
    uEmissive:        { value: lerpState.emissive },
    uTime:            { value: 0 },
    uNoiseAmp:        { value: 0 },
    uNoiseAmp2:       { value: 0 },
    uNoiseFreq:       { value: 0 },
    uNoiseFreq2:      { value: 0 },
    uPulse:           { value: 0 },
    uBounce:          { value: 0 },
    uRippleOrigins:   { value: rippleOriginsArr },
    uRippleTimes:     { value: rippleTimesArr },
    uRippleStrengths: { value: rippleStrengthsArr },
    uRippleCount:     { value: 0 },
  },
  transparent: true,
});

const blob = new THREE.Mesh(geo, mat);
scene.add(blob);

// ── Glow Sphere ─────────────────────────────────────────────────────
const glowGeo = new THREE.SphereGeometry(1.8, 64, 64);
const glowMat = new THREE.ShaderMaterial({
  vertexShader: /* glsl */ `
    varying vec3 vNormal;
    void main() {
      vNormal     = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform vec3 uGlowColor;
    varying vec3 vNormal;
    void main() {
      float i = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
      gl_FragColor = vec4(uGlowColor, i * 0.2);
    }
  `,
  uniforms: { uGlowColor: { value: lerpState.colorA.clone() } },
  transparent: true,
  side: THREE.BackSide,
  depthWrite: false,
});
scene.add(new THREE.Mesh(glowGeo, glowMat));

// ── Floating Particles ──────────────────────────────────────────────
const PARTICLE_COUNT = 60;
const particleGeo = new THREE.BufferGeometry();
const pPositions = new Float32Array(PARTICLE_COUNT * 3);
const particleSpeeds = [];

for (let i = 0; i < PARTICLE_COUNT; i++) {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = 2.0 + Math.random() * 2.0;
  pPositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
  pPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
  pPositions[i * 3 + 2] = r * Math.cos(phi);
  particleSpeeds.push({
    speed:  0.1 + Math.random() * 0.3,
    offset: Math.random() * Math.PI * 2,
    radius: r,
  });
}

particleGeo.setAttribute("position", new THREE.BufferAttribute(pPositions, 3));

const particleMat = new THREE.PointsMaterial({
  color: 0xc4b0ff,
  size: 0.02,
  transparent: true,
  opacity: 0.4,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const particles = new THREE.Points(particleGeo, particleMat);
scene.add(particles);

// =====================================================================
//  RAYCASTER – localized ripple + whole-body spring bounce
// =====================================================================
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function onPointerDown(e) {
  const x = (e.clientX !== undefined) ? e.clientX : (e.touches && e.touches[0].clientX);
  const y = (e.clientY !== undefined) ? e.clientY : (e.touches && e.touches[0].clientY);
  if (x === undefined) return;

  pointer.x = (x / window.innerWidth) * 2 - 1;
  pointer.y = -(y / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(blob);

  if (hits.length > 0) {
    // Direct hit: spawn localized ripple at hit point on unit sphere
    const localPt = hits[0].point.clone();
    blob.worldToLocal(localPt);
    localPt.normalize();

    if (ripples.length >= MAX_RIPPLES) ripples.shift();
    ripples.push({ origin: localPt, time: 0, strength: 0.25 });

    // Kick the whole blob outward then spring back
    bounceVelocity += 0.08;
  } else {
    // Background click: gentle bounce only
    bounceVelocity += 0.04;
  }
}

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("touchstart", onPointerDown, { passive: true });

// =====================================================================
//  ANIMATION LOOP
// =====================================================================
let time = 0;
const LERP_SPEED = 0.03;

function lerpVal(a, b, t) {
  return a + (b - a) * t;
}

function animate() {
  requestAnimationFrame(animate);
  time += 0.012; // fast enough for visible flowing motion

  // ── Intensity scaling ───────────────────────────────────────────
  const iFactor = 0.3 + intensity * 0.7;

  // ── Lerp toward target state ────────────────────────────────────
  lerpState.speed     = lerpVal(lerpState.speed,     targetState.speed     * iFactor, LERP_SPEED);
  lerpState.noiseAmp  = lerpVal(lerpState.noiseAmp,  targetState.noiseAmp  * iFactor, LERP_SPEED);
  lerpState.noiseAmp2 = lerpVal(lerpState.noiseAmp2, targetState.noiseAmp2 * iFactor, LERP_SPEED);
  lerpState.noiseFreq = lerpVal(lerpState.noiseFreq, targetState.noiseFreq * iFactor, LERP_SPEED);
  lerpState.noiseFreq2 = lerpVal(lerpState.noiseFreq2, targetState.noiseFreq2 * iFactor, LERP_SPEED);
  lerpState.rotSpeed  = lerpVal(lerpState.rotSpeed,  targetState.rotSpeed  * iFactor, LERP_SPEED);
  lerpState.pulseAmp  = lerpVal(lerpState.pulseAmp,  targetState.pulseAmp  * iFactor, LERP_SPEED);
  lerpState.pulseFreq = lerpVal(lerpState.pulseFreq, targetState.pulseFreq * iFactor, LERP_SPEED);
  lerpState.colorA.lerp(targetState.colorA, LERP_SPEED);
  lerpState.colorB.lerp(targetState.colorB, LERP_SPEED);
  lerpState.emissive.lerp(targetState.emissive, LERP_SPEED);

  // ── Spring-bounce physics ───────────────────────────────────────
  bounceVelocity += bounceImpulse * BOUNCE_SPRING * 0.016;
  bounceVelocity *= BOUNCE_DAMPING;
  bounceImpulse += bounceVelocity * 0.016;
  if (Math.abs(bounceImpulse) < 0.0005 && Math.abs(bounceVelocity) < 0.0005) {
    bounceImpulse = 0;
    bounceVelocity = 0;
  }

  // ── Ripple aging ────────────────────────────────────────────────
  for (let i = ripples.length - 1; i >= 0; i--) {
    ripples[i].time += 0.016;
    if (ripples[i].time > 3.0) ripples.splice(i, 1);
  }

  // ── Upload ripple data to uniforms ──────────────────────────────
  const rCount = Math.min(ripples.length, MAX_RIPPLES);
  for (let i = 0; i < MAX_RIPPLES; i++) {
    if (i < rCount) {
      rippleOriginsArr[i].copy(ripples[i].origin);
      rippleTimesArr[i] = ripples[i].time;
      rippleStrengthsArr[i] = ripples[i].strength;
    } else {
      rippleTimesArr[i] = 0;
      rippleStrengthsArr[i] = 0;
    }
  }
  mat.uniforms.uRippleCount.value = rCount;

  // ── Pulse (breathing) ───────────────────────────────────────────
  const pulse = Math.sin(time * lerpState.pulseFreq) * lerpState.pulseAmp;

  // ── Update uniforms ─────────────────────────────────────────────
  mat.uniforms.uColorA.value.copy(lerpState.colorA);
  mat.uniforms.uColorB.value.copy(lerpState.colorB);
  mat.uniforms.uEmissive.value.copy(lerpState.emissive);
  mat.uniforms.uTime.value         = time * lerpState.speed;
  mat.uniforms.uNoiseAmp.value     = lerpState.noiseAmp;
  mat.uniforms.uNoiseAmp2.value    = lerpState.noiseAmp2;
  mat.uniforms.uNoiseFreq.value    = lerpState.noiseFreq;
  mat.uniforms.uNoiseFreq2.value   = lerpState.noiseFreq2;
  mat.uniforms.uPulse.value        = pulse;
  mat.uniforms.uBounce.value       = bounceImpulse;
  glowMat.uniforms.uGlowColor.value.copy(lerpState.colorA);

  // ── Rotate (asymmetric for organic feel) ────────────────────────
  blob.rotation.x += 0.0015 * lerpState.rotSpeed;
  blob.rotation.y += 0.0025 * lerpState.rotSpeed;
  blob.rotation.z += 0.0008 * lerpState.rotSpeed;

  // ── Animate particles ───────────────────────────────────────────
  const pArr = particleGeo.attributes.position.array;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const s = particleSpeeds[i];
    const angle = time * s.speed * iFactor + s.offset;
    const r = s.radius + Math.sin(time * 0.3 + s.offset) * 0.2;
    pArr[i * 3]     = r * Math.sin(angle) * Math.cos(angle * 0.7);
    pArr[i * 3 + 1] = r * Math.cos(angle) * Math.sin(angle * 0.3);
    pArr[i * 3 + 2] = r * Math.sin(angle * 0.5) * Math.cos(angle * 0.9);
  }
  particleGeo.attributes.position.needsUpdate = true;
  particleMat.color.copy(lerpState.colorA);

  // ── Gentle camera bob ───────────────────────────────────────────
  camera.position.y = Math.sin(time * 0.2) * 0.03;

  renderer.render(scene, camera);
}

// ── Resize ──────────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Streamlit Component Protocol ────────────────────────────────────
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "streamlit:render") {
    const args = event.data.args;
    if (args && args.state) setState(args.state);
    if (args && args.intensity !== undefined) setBlobIntensity(args.intensity);
  }
});

(function streamlitInit() {
  const msg = { isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 };
  window.parent.postMessage(msg, "*");
  const heightMsg = { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: 800 };
  window.parent.postMessage(heightMsg, "*");
})();

// ── Start ───────────────────────────────────────────────────────────
animate();
