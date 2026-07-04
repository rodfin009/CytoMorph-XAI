# CytoMorph-XAI: Production-Grade Automated WBC Classifier with Explainable AI (XAI)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ed.svg)](https://www.docker.com/)

An end-to-end, production-ready Deep Learning microservice engineered to automate White Blood Cell (WBC) differential counting from digital microscopic imagery. Powered by **EfficientNet-B0** and wrapped in an asynchronous **FastAPI** web server, the system achieves **99% Macro Accuracy** while implementing **Grad-CAM** for clinical transparency (Explainable AI).

---

## 🔬 The Clinical Challenge & AI Solution

Manual WBC differential counting is a time-consuming, labor-intensive process prone to human fatigue and inter-observer variability. While automated hematology analyzers provide absolute cell counts, morphology validation still requires expert microscopic review.

**CytoMorph-XAI** bridges this gap by offering a fast, stable, and highly generalizable computer vision pipeline that acts as a clinical decision support system. It accurately classifies leukocytes into their five primary physiological lineages: **Basophils, Eosinophils, Lymphocytes, Monocytes, and Neutrophils.**

---

## 🛠️ System Architecture Diagram

```text
[Digital Microscope Image] 
           │
           ▼
[Preprocessing & Augmentation Pipeline] (Albumentations, ColorJitter)
           │
           ▼
[EfficientNet-B0 Backbone] (Pre-trained on ImageNet)
           │
           ▼
[Custom Clinical Classifier Head] (Dropout, Fully Connected Layers)
           │
           ▼
[Explainable AI Layer] (Grad-CAM Activation Mapping)
           │
           ▼
[FastAPI Asynchronous Gateway] ───► [Bilingual JSON Response & Heatmap]
