# CyberSentiment - Cybersecurity Threat Intelligence Platform

## Project Overview

CyberSentiment is a comprehensive web application designed to monitor, analyze, and alert on cybersecurity-related discussions across social media platforms. The system automatically scans Twitter and Reddit for cybersecurity topics, analyzes the sentiment and risk level of posts, and provides a dashboard for security analysts to monitor potential threats in real-time.

This application serves as an early warning system that helps cybersecurity professionals stay informed about emerging threats, data breaches, ransomware attacks, and other security incidents being discussed on social media. It combines automated data collection with machine learning analysis to identify potentially risky content before it becomes widespread.

## Live Application

The application is currently live and accessible at:
**https://cybersentiment.pythonanywhere.com/**

## Technology Stack

### Backend Development
- **Flask** - Lightweight Python web framework for application core
- **Flask-SQLAlchemy** - Database ORM and management
- **Flask-Login** - User authentication and session management
- **Flask-WTF** - Form handling and validation
- **Flask-Migrate** - Database migration management

### Data Science & Machine Learning
- **Scikit-learn** - Machine learning models for sentiment classification
- **NLTK** - Natural language processing and text analysis
- **VADER Sentiment Analysis** - Lexicon-based sentiment scoring
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing operations
- **Joblib** - Model serialization and loading

### Data Collection & APIs
- **Tweepy** - Twitter API integration for data collection
- **PRAW** - Reddit API integration for data collection
- **Requests** - HTTP library for API communications
- **Python-dotenv** - Environment variable management for API keys

### Database Systems
- **SQLite** - Local development database
- **MySQL** - Production database on PythonAnywhere
- **SQLAlchemy** - Database abstraction and query management

### Frontend & User Interface
- **HTML5** - Page structure and semantic markup
- **CSS3** - Styling and responsive design
- **Bootstrap 5** - Frontend framework for responsive layout
- **JavaScript** - Client-side interactivity
- **Chart.js** - Data visualization and analytics charts
- **Jinja2** - Server-side templating engine

### Deployment & Infrastructure
- **PythonAnywhere** - Cloud hosting platform
- **Virtual Environment** - Python dependency isolation
- **MySQL Connector Python** - Database connectivity
- **Cron Scheduling** - Automated daily data collection

### Security & Authentication
- **Flask-Login** - User session management
- **Werkzeug** - Password hashing and security utilities
- **Secure Session Management** - Encrypted user sessions
- **Environment Variables** - Secure credential management

## Key Features

### Automated Social Media Monitoring
- Continuous scanning of Twitter and Reddit for cybersecurity-related discussions
- Targeted keyword searches including "ransomware," "phishing," "data breach," "cyberattack," and other security terms
- Daily automated data collection from multiple subreddits and Twitter streams
- Real-time processing of new posts as they appear

### Advanced Sentiment Analysis
- Dual-layer sentiment analysis using both VADER lexicon-based scoring and machine learning classification
- Automated risk assessment to flag potentially dangerous content
- Sentiment trend tracking over time with visual analytics
- Confidence scoring for each sentiment prediction

### Intelligent Alert System
- Automatic alert generation for high-risk posts
- Personalized notification management for each user
- Visual indicators for read and unread alerts
- Bulk alert management capabilities

### User-Friendly Dashboard
- Comprehensive overview of threat landscape with key metrics
- Interactive sentiment trend charts and analytics
- Quick access to latest alerts and notifications
- Search and filter capabilities across all collected data

### Secure User Management
- User registration and authentication system
- Personalized alert tracking and management
- Secure session management
- Role-based access control

## Technical Architecture

### Backend Framework
- Built using Flask web framework for lightweight and efficient performance
- SQLite database for local development and testing
- MySQL database for production deployment on PythonAnywhere
- SQLAlchemy ORM for database management and migrations

### Data Processing Pipeline
- Automated daily scraping of Twitter and Reddit data
- Text preprocessing and cleaning pipeline
- Machine learning model integration for sentiment classification
- Real-time risk assessment and alert generation

### Deployment Environment
- Hosted on PythonAnywhere cloud platform
- Automated daily data collection via scheduled tasks
- Virtual environment for package management
- Production-grade MySQL database

## User Guide

### Getting Started
1. Visit the live application at https://cybersentiment.pythonanywhere.com/
2. Register for a new account or login with existing credentials
3. Access the dashboard for an overview of current threat intelligence

### Dashboard Features
- View total analyzed posts and sentiment distribution
- Monitor negative sentiment alerts that may indicate emerging threats
- Access latest alerts and notifications in real-time
- Track sentiment trends over customizable time periods

### Post Management
- Browse all collected posts with pagination
- Search posts by keywords, platform, or sentiment
- View detailed post analysis including sentiment scores
- Flag posts for manual review and attention

### Alert System
- Receive automatic alerts for high-risk content
- Mark alerts as read to track reviewed content
- Access comprehensive alert history
- Use bulk operations for efficient alert management

### Analytics and Reporting
- Monitor sentiment trends across different time frames
- Analyze platform-specific threat patterns
- Track risk probability distributions
- Generate insights from aggregated sentiment data

## System Benefits

### For Security Teams
- Early detection of emerging cyber threats
- Real-time monitoring of threat actor discussions
- Reduced time to discovery for security incidents
- Complementary intelligence to traditional security tools

### For Security Analysts
- Consolidated view of social media threat intelligence
- Automated sorting and prioritization of potential threats
- Historical trend analysis for pattern recognition
- Customizable alerting based on risk tolerance

### For Organizations
- Enhanced situational awareness of cyber threat landscape
- Proactive risk management through early warning indicators
- Cost-effective threat intelligence gathering
- Scalable monitoring solution for growing security needs

## Maintenance and Support

The system is designed for continuous operation with:
- Automated daily data collection and processing
- Regular database maintenance and optimization
- Continuous monitoring of system health
- Periodic model updates for improved accuracy

## Security Considerations

- Secure user authentication and session management
- Encrypted database connections
- Regular security updates for all dependencies
- Secure API key management for social media platforms

This platform represents a complete operational system that bridges social media monitoring with cybersecurity threat intelligence, providing actionable insights through an accessible web interface for security professionals worldwide. The technology stack has been carefully selected to provide robust performance while maintaining scalability and ease of maintenance.