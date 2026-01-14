import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr
import math

# Direct imports to bypass attribute errors
import mediapipe.solutions.pose as mp_pose
import mediapipe.solutions.drawing_utils as mp_drawing
import mediapipe.solutions.drawing_styles as mp_drawing_styles

# Data Structures
@dataclass
class Landmark:
    x: float; y: float; z: float; visibility: float

@dataclass
class PoseData:
    landmarks: List[Landmark]
    world_landmarks: List[Landmark]
    timestamp: float
    frame_index: int
    raw_landmarks: any

@dataclass
class MetricData:
    knee_angle: float; trunk_lean: float; hip_shoulder_sync: float
    phase: str; heel_lift: bool; frame_index: int

@dataclass
class RepSummary:
    rep_index: int; start_frame: int; end_frame: int
    depth_score: float; trunk_score: float; heel_score: float
    sync_score: float; eccentric_tempo: float; concentric_tempo: float

@dataclass
class SetSummary:
    reps: List[RepSummary]
    overall_score: float
    letter_grade: str
    radar_scores: Dict[str, float]
    raw_averages: Dict[str, float]

class PoseProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        all_pose_data = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks = [Landmark(l.x, l.y, l.z, l.visibility) for l in results.pose_landmarks.landmark]
                world_landmarks = [Landmark(l.x, l.y, l.z, l.visibility) for l in results.pose_world_landmarks.landmark]
                all_pose_data.append(PoseData(landmarks, world_landmarks, frame_idx/fps, frame_idx, results.pose_landmarks))
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
        cap.release()
        return all_pose_data, fps

class BiomechanicalAnalyzer:
    def __init__(self):
        self.L_HIP, self.L_KNEE, self.L_ANKLE, self.L_SHOULDER = 23, 25, 27, 11
        self.L_HEEL, self.L_FOOT_INDEX = 29, 31 

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def analyze(self, pose_list, fps):
        metrics = []
        for p in pose_list:
            w, l = p.world_landmarks, p.landmarks
            knee = self._calculate_angle(w[self.L_HIP], w[self.L_KNEE], w[self.L_ANKLE])
            trunk_vec = np.array([w[self.L_SHOULDER].x - w[self.L_HIP].x, w[self.L_SHOULDER].y - w[self.L_HIP].y])
            trunk_lean = np.degrees(np.arccos(np.dot(trunk_vec, [0, -1]) / np.linalg.norm(trunk_vec)))
            heel_lift = l[self.L_HEEL].y < (l[self.L_FOOT_INDEX].y - 0.01)
            metrics.append(MetricData(knee, trunk_lean, 0.0, 'START', heel_lift, p.frame_index))

        angles_series = pd.Series([m.knee_angle for m in metrics]).rolling(5, center=True).mean()
        angles = angles_series.bfill().ffill().values
        for i in range(1, len(metrics)):
            diff = angles[i] - angles[i-1]
            if angles[i] > 160: phase = "START"
            elif diff < -0.5: phase = "DESCENT"
            elif abs(diff) < 0.2 and metrics[i-1].phase == "DESCENT": phase = "BOTTOM"
            elif diff > 0.5: phase = "ASCENT"
            else: phase = metrics[i-1].phase
            metrics[i].phase = phase

        reps, in_rep, start_idx = [], False, 0
        for i, m in enumerate(metrics):
            if m.phase == "DESCENT" and not in_rep: in_rep, start_idx = True, i
            if m.phase == "START" and in_rep:
                rep_frames = pose_list[start_idx:i]
                rep_m = metrics[start_idx:i]
                if len(rep_m) > 15:
                    ascent_indices = [idx for idx, rm in enumerate(rep_m) if rm.phase == "ASCENT"]
                    if len(ascent_indices) > 5:
                        hip_v = np.diff([rep_frames[idx].landmarks[self.L_HIP].y for idx in ascent_indices])
                        sho_v = np.diff([rep_frames[idx].landmarks[self.L_SHOULDER].y for idx in ascent_indices])
                        corr, _ = pearsonr(hip_v, sho_v)
                        sync_score = max(0, corr * 100)
                    else: sync_score = 75.0
                    reps.append(RepSummary(len(reps), start_idx, i, min([rm.knee_angle for rm in rep_m]), max([rm.trunk_lean for rm in rep_m]), 0.0 if any([rm.heel_lift for rm in rep_m]) else 100.0, sync_score, len([rm for rm in rep_m if rm.phase=="DESCENT"])/fps, len([rm for rm in rep_m if rm.phase=="ASCENT"])/fps))
                in_rep = False
        return metrics, reps

class GradingEngine:
    def grade(self, reps):
        if not reps: return SetSummary([], 0, "F", {}, {})
        raw = {'Depth': np.mean([r.depth_score for r in reps]), 'Trunk': np.mean([r.trunk_score for r in reps]), 'Ecc Tempo': np.mean([r.eccentric_tempo for r in reps]), 'Conc Tempo': np.mean([r.concentric_tempo for r in reps]), 'Sync': np.mean([r.sync_score for r in reps]), 'Heel Lifts': sum([1 for r in reps if r.heel_score == 0])}
        scores = {'Depth': np.mean([min(100, ((180-r.depth_score)/90)*100) for r in reps]), 'Trunk': np.mean([max(0, 100 - ((r.trunk_score-20)/30*100)) for r in reps]), 'Heel': np.mean([r.heel_score for r in reps]), 'Sync': raw['Sync'], 'Ecc Tempo': np.mean([100 if 2<=r.eccentric_tempo<=4 else 50 for r in reps]), 'Conc Tempo': np.mean([100 if r.concentric_tempo<=3 else 50 for r in reps])}
        weights = {'Depth': 0.3, 'Trunk': 0.25, 'Heel': 0.15, 'Sync': 0.15, 'Ecc Tempo': 0.1, 'Conc Tempo': 0.05}
        overall = sum(scores[k] * weights[k] for k in weights)
        grade = 'A' if overall>=90 else 'B' if overall>=80 else 'C' if overall>=70 else 'D' if overall>=60 else 'F'
        return SetSummary(reps, overall, grade, scores, raw)

# UI Logic
st.set_page_config(page_title="Squat AI Pro", layout="wide")
st.title("Squat Form AI Analyzer")
uploaded_file = st.sidebar.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    temp_input_path = "input.mp4"
    temp_output_path = "output_raw.mp4"
    final_web_path = "output_web.mp4"

    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    processor = PoseProcessor()
    pose_data, fps = processor.process_video(temp_input_path)
    analyzer = BiomechanicalAnalyzer()
    metrics, reps = analyzer.analyze(pose_data, fps)
    summary = GradingEngine().grade(reps)

    with col1:
        st.subheader("Results")
        # Radar Chart
        categories = list(summary.radar_scores.keys())
        values = list(summary.radar_scores.values()) + [list(summary.radar_scores.values())[0]]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist() + [0]
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='teal', alpha=0.3)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories)
        st.pyplot(fig)
        
        raw = summary.raw_averages
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"Depth: {int(raw['Depth'])} deg"); st.write(f"Trunk: {int(raw['Trunk'])} deg")
        with c2:
            st.write(f"Down: {raw['Ecc Tempo']:.1f}s"); st.write(f"Up: {raw['Conc Tempo']:.1f}s")
        with c3:
            st.write(f"Sync: {int(raw['Sync'])}/100"); st.write(f"Heels: {int(raw['Heel Lifts'])} lifts")

    with col2:
        st.subheader("Video")
        cap = cv2.VideoCapture(temp_input_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        pose_map = {p.frame_index: p for p in pose_data}
        metric_map = {m.frame_index: m for m in metrics}

        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret: break
            if i in pose_map:
                mp_drawing.draw_landmarks(frame, pose_map[i].raw_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing_styles.get_default_pose_landmarks_style())
            if i in metric_map:
                cv2.putText(frame, f"Phase: {metric_map[i].phase}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            out.write(frame)
        cap.release(); out.release()

        # CONVERT TO WEB-FRIENDLY H.264
        if os.path.exists(temp_output_path):
            os.system(f"ffmpeg -i {temp_output_path} -vcodec libx264 {final_web_path} -y")
            st.video(final_web_path)
