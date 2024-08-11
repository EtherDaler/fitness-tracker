# Fitness Tracker

A fitness tracker application that allows users to record themselves and track their workout progress over time.
This project is built with FastAPI, PostgreSQL and MediaPipe.

## Features

### 1. User authentication

Authorization with JWT.

### 2. Emailing

Send registeration emails and links.

### 3. Flexibility

User can create multiple sessions of the same exercise and track them individually.

### 4. AI Workout Tracking

The workout is tracked using AI models. The user will open their camera and the app will send the live footage in the backend server, where ML models will calculate how accurate the exercise is and the frotnend will display the repetition count.

### 3. Workout Data

The user's activity will be stored, various mathematical calculations will be done on the data and the user will be shown the progress in the frontend in the charts.

## Structure

```
.
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── api
│   │   └── v1
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── exercises.py
│   │       ├── tokens.py
│   │       ├── users.py
│   │       ├── websoc-kets.py
│   │       └── workouts.py
│   ├── config.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── emails.py
│   │   ├── security.py
│   │   └── utils.py
│   ├── dependencies
│   │   ├── __init__.py
│   │   ├── jwt.py
│   │   └── workouts.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── auth_token.py
│   │   ├── exercises.py
│   │   ├── users.py
│   │   ├── work_session.py
│   │   └── workouts.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── exercises.py
│   │   ├── tokens.py
│   │   ├── users.py
│   │   └── workouts.py
│   ├── static
│   │   ├── audio
│   │   │   └── *.mp3
│   │   ├── css
│   │   │   ├── main.css
│   │   │   └── tailwind.css
│   │   ├── favicon.ico
│   │   ├── imgs
│   │   │   └── *.png
│   │   ├── js
│   │   │   ├── lib.js
│   │   │   ├── login.js
│   │   │   ├── newWorkouts.js
│   │   │   ├── profile.js
│   │   │   ├── register.js
│   │   │   └── store.js
│   │   ├── uploads
│   │   │   ├── *.png
│   │   └── videos
│   │       ├── high_knees.mov
│   │       └── jumping_jacks.mp4
│   ├── styles
│   │   └── main.css
│   └── templates
│       ├── dashboard.html
│       ├── emails
│       │   └── temporary-password.html
│       ├── index.html
│       ├── login.html
│       ├── new-workouts.html
│       ├── profile.html
│       ├── register.html
│       └── workout.html
├── requirements.txt
├── runtime.txt
└── tailwind.config.js
```
