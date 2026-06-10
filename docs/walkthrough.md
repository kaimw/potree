# Fix: HQ + Orthographic Rendering Disappears on Android Tablets

## Problem
Selecting **High Quality** + **Orthographic** camera mode in Potree causes the rendered point cloud to disappear. The bug was previously fixed on PC, but persisted on Android tablets.

## Root Cause Analysis

The issue stems from **NaN value propagation** which mobile GPUs (Adreno, Mali) handle differently from desktop GPUs:

### 1. NaN `fov` Uniform
In [PotreeRenderer.js](file:///Users/kaimao/github/potree/src/PotreeRenderer.js#L1236), `OrthographicCamera` has no `fov` property, so:
```javascript
shader.setUniform1f("fov", Math.PI * camera.fov / 180);
// camera.fov = undefined → Math.PI * undefined / 180 = NaN
```

### 2. NaN Leaks Through Shader Branches on Mobile GPUs
In [pointcloud.vs](file:///Users/kaimao/github/potree/src/materials/shaders/pointcloud.vs#L669), the original code computed `tan(NaN)` **unconditionally** before the `if/else` branch:
```glsl
float slope = tan(fov / 2.0);  // NaN when fov is NaN — always computed!
if (uUseOrthographicCamera) {
    projFactor = uScreenWidth / uOrthoWidth;  // valid value, but...
} else {
    projFactor = -0.5 * uScreenHeight / (slope * vViewPosition.z);  // NaN
}
```
On desktop GPUs, the unused NaN is harmlessly discarded. **Mobile GPUs often execute both branches** (SIMD architecture) and NaN values in registers can contaminate subsequent operations.

### 3. `vLogDepth` NaN
`log2(-mvPosition.z)` produces NaN when `mvPosition.z ≥ 0` (valid in orthographic view), corrupting the EDL depth used by the HQ normalization pass.

## Changes Made

### [PotreeRenderer.js](file:///Users/kaimao/github/potree/src/PotreeRenderer.js#L1236)
Set a safe fallback for `fov` uniform when camera has no `fov` property:
```diff
-shader.setUniform1f("fov", Math.PI * camera.fov / 180);
+shader.setUniform1f("fov", camera.fov ? Math.PI * camera.fov / 180 : 1.0);
```

### [pointcloud.vs](file:///Users/kaimao/github/potree/src/materials/shaders/pointcloud.vs#L669-L675)
Moved `slope` computation inside the `else` branch to prevent any NaN computation:
```diff
-float slope = tan(fov / 2.0);
 float projFactor;
 if (uUseOrthographicCamera) {
     projFactor = uScreenWidth / uOrthoWidth;
 } else {
+    float slope = tan(fov / 2.0);
     projFactor = -0.5 * uScreenHeight / (slope * vViewPosition.z);
 }
```

### [pointcloud.vs](file:///Users/kaimao/github/potree/src/materials/shaders/pointcloud.vs#L871) (main function)
Clamped `vLogDepth` to prevent NaN from `log2` of negative/zero values:
```diff
-vLogDepth = log2(-mvPosition.z);
+vLogDepth = log2(max(1e-7, -mvPosition.z));
```

## Verification
- Built the project successfully with `npm run build`
- Tested on PC browser: HQ + Orthographic renders correctly

![HQ + Orthographic rendering after fix](file:///Users/kaimao/.gemini/antigravity/brain/475a089d-02c2-4458-a1ea-ef43cf577420/lion_hq_ortho_fixed.png)

> [!IMPORTANT]
> Please test on the Android tablet to confirm the fix works. The changes eliminate all NaN value paths in the shader, which was the root cause of the mobile GPU rendering failure.
