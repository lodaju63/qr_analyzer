# GitHub Actions로 Mac 앱 빌드하기

## 🎯 **개요**

Windows PC에서도 GitHub Actions를 사용해서 **Mac용 .app 파일을 자동으로 빌드**할 수 있습니다!

---

## 📋 **사전 준비**

### 1️⃣ **GitHub 계정**
- GitHub 계정 생성 (무료)
- Repository 생성

### 2️⃣ **Git 설치**
```bash
# Windows
winget install Git.Git

# 또는 https://git-scm.com/download/win
```

---

## 🚀 **사용 방법**

### 1️⃣ **Git 초기화 및 GitHub에 Push**

```bash
# 프로젝트 폴더에서
cd c:\Users\Administrator\qr_sh

# Git 초기화 (처음 1회)
git init
git add .
git commit -m "Initial commit - QR Analyzer"

# GitHub에 연결 (본인의 리포지토리 URL 사용)
git remote add origin https://github.com/YOUR_USERNAME/qr_analyzer.git
git branch -M main
git push -u origin main
```

### 2️⃣ **GitHub Actions 자동 실행**

Push하면 **자동으로** 빌드 시작!

```
GitHub 사이트 → Your Repository → Actions 탭
→ "Build All Platforms" 워크플로우 확인
```

### 3️⃣ **빌드 결과 다운로드**

```
Actions 탭 → 완료된 워크플로우 클릭 → Artifacts
→ "QR_Analyzer_Mac" 다운로드! (Mac .app + .dmg)
→ "QR_Analyzer_Windows" 다운로드! (Windows .exe)
```

---

## 🔧 **워크플로우 파일**

생성된 파일들:

### 1️⃣ **`.github/workflows/build-mac.yml`**
- Mac 전용 빌드
- `QR_Analyzer.app` 생성
- `QR_Analyzer.dmg` 생성

### 2️⃣ **`.github/workflows/build-all.yml`**
- Windows + Mac 동시 빌드
- 두 플랫폼 모두 자동 빌드
- Release 자동 생성 (태그 시)

---

## 🎯 **빌드 트리거**

### **자동 빌드**
```bash
# main 브랜치에 push하면 자동 빌드
git add .
git commit -m "Update app"
git push
```

### **수동 빌드**
```
GitHub 사이트 → Actions → Build All Platforms
→ "Run workflow" 버튼 클릭!
```

### **Release 생성**
```bash
# 버전 태그 추가
git tag v1.0.0
git push origin v1.0.0

# 자동으로 Release 생성 + 파일 첨부!
```

---

## 📦 **빌드 프로세스**

### **GitHub Actions Runner**
```
1. Push 감지
2. macOS 가상 머신 시작
3. Python 3.10 설치
4. 의존성 설치 (PyTorch, PyQt6, etc.)
5. PyInstaller 실행
6. .app 및 .dmg 생성
7. Artifact 업로드
```

**소요 시간:** 약 15-20분

---

## 🎉 **장점**

### ✅ **편리함**
- Windows에서 작업해도 Mac 앱 빌드 가능!
- Mac 실제 장비 불필요!
- 자동화!

### ✅ **무료**
- GitHub Actions 무료 (월 2000분)
- Public 리포지토리면 무제한

### ✅ **자동화**
- Push만 하면 자동 빌드
- 여러 플랫폼 동시 빌드
- Release 자동 생성

---

## 📋 **Artifacts 다운로드**

### 1️⃣ **GitHub 웹사이트**
```
Repository → Actions → 완료된 워크플로우
→ Artifacts 섹션 → 다운로드 ZIP
```

### 2️⃣ **ZIP 압축 해제**
```
QR_Analyzer_Mac.zip 압축 해제
→ QR_Analyzer.app
→ QR_Analyzer.dmg
```

### 3️⃣ **배포**
```
.app 또는 .dmg 파일을 사용자에게 전달!
```

---

## ⚠️ **주의사항**

### 1️⃣ **GitHub 용량 제한**
- 단일 파일: 최대 2GB
- Artifact: 7일 보관 (기본)

### 2️⃣ **빌드 시간**
- Mac 빌드: 15-20분
- Windows 빌드: 10-15분
- 무료 계정: 월 2000분 제한

### 3️⃣ **Private vs Public**
- **Public 리포지토리**: 무제한 빌드
- **Private 리포지토리**: 월 2000분 제한

---

## 🔐 **민감 정보 처리**

### **비밀번호 하드코딩 주의!**

`main.py`에서:
```python
CORRECT_PASSWORD = "2017112166"  # 하드코딩됨!
```

### **GitHub Secrets 사용 (선택사항)**
1. Repository → Settings → Secrets → New secret
2. Name: `APP_PASSWORD`
3. Value: `2017112166`
4. 코드에서 환경 변수로 읽기

---

## 🎯 **사용 예시**

### **시나리오 1: Mac 앱만 빌드**
```bash
git add .
git commit -m "Build Mac app"
git push

# GitHub Actions → build-mac.yml 실행
# 15분 후 → QR_Analyzer_Mac.zip 다운로드!
```

### **시나리오 2: Windows + Mac 동시 빌드**
```bash
git add .
git commit -m "Build all platforms"
git push

# GitHub Actions → build-all.yml 실행
# 20분 후:
# - QR_Analyzer_Windows.zip
# - QR_Analyzer_Mac.zip
# 둘 다 다운로드 가능!
```

### **시나리오 3: Release 생성**
```bash
git tag v1.0.0
git push origin v1.0.0

# 자동으로:
# 1. Windows .exe 빌드
# 2. Mac .dmg 빌드
# 3. GitHub Release 생성
# 4. 파일 자동 첨부!
```

---

## 📊 **워크플로우 상태 확인**

### **실시간 로그 보기**
```
GitHub → Actions → 진행 중인 워크플로우 클릭
→ "build-mac" 또는 "build-windows" 클릭
→ 각 단계별 로그 확인!
```

### **빌드 성공 여부**
```
✅ 녹색 체크: 빌드 성공
❌ 빨간 X: 빌드 실패 (로그 확인)
🟡 노란 원: 빌드 진행 중
```

---

## 💡 **빠른 시작 가이드**

### **1분 만에 Mac 앱 빌드!**

```bash
# 1. GitHub에 리포지토리 생성
# 2. 로컬에서 Git 설정
cd c:\Users\Administrator\qr_sh
git init
git add .
git commit -m "Add QR Analyzer"
git remote add origin https://github.com/YOUR_USERNAME/qr_analyzer.git
git push -u origin main

# 3. GitHub Actions 자동 실행!
# 4. 15분 후 Artifacts에서 다운로드!
```

---

## 🎉 **완료!**

### ✅ **Windows PC에서 할 수 있는 것**
- Mac 앱 빌드 (.app, .dmg)
- Windows 앱 빌드 (.exe)
- 자동화된 빌드 프로세스
- Release 관리

### ✅ **GitHub Actions 제공**
- macOS 가상 머신
- Windows 가상 머신
- 무료 빌드 (Public 리포지토리)
- Artifact 자동 업로드

---

## 📧 **다음 단계**

1. GitHub에 리포지토리 생성
2. 코드 Push
3. Actions 탭에서 빌드 확인
4. Artifacts 다운로드
5. 배포!

---

**🚀 GitHub Actions로 Mac 앱도 Windows에서 빌드 가능! 🎊**
