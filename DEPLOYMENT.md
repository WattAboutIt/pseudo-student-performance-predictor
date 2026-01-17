# Deploying to Render (Free)

This guide will help you deploy the Student Performance Predictor to Render for free.

## Prerequisites

- GitHub account
- Render account (free at https://render.com)

## Step 1: Push Code to GitHub

1. **Initialize a git repository** (if not already done):

   ```bash
   cd "/home/mushroom/Student Performance Predictor"
   git init
   git add .
   git commit -m "Initial commit: Student Performance Predictor"
   ```

2. **Create a GitHub repository**:
   - Go to https://github.com/new
   - Create a repository (e.g., `student-performance-predictor`)
   - Don't initialize with README (you already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/student-performance-predictor.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy on Render

1. **Go to Render**:
   - Visit https://dashboard.render.com
   - Sign up (free) with GitHub

2. **Connect GitHub**:
   - Click "New +" → "Web Service"
   - Click "Connect repository"
   - Authorize GitHub if prompted
   - Select your `student-performance-predictor` repository

3. **Configure the Service**:
   - **Name**: `student-performance-predictor` (or any name)
   - **Environment**: `Python 3`
   - **Region**: `Oregon` (or closest to you)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`

4. **Click "Create Web Service"**

5. **Wait for deployment**:
   - Render will:
     - Install dependencies
     - Train the model (takes a few minutes)
     - Start the app
   - Check the logs to see progress

## Step 3: Access Your App

Once deployed, you'll get a URL like: `https://student-performance-predictor.onrender.com`

- **Web UI**: https://student-performance-predictor.onrender.com
- **API Docs**: https://student-performance-predictor.onrender.com/docs
- **API Health**: https://student-performance-predictor.onrender.com/api/health

## Important Notes

### Free Plan Limitations

✅ **What you get for free:**

- 0.5 GB RAM
- Shared CPU
- Automatic deploys from GitHub
- HTTPS SSL
- Up to 750 free tier hours/month (enough for continuous uptime)

⚠️ **Considerations:**

- App will be slow during first request after inactivity (cold start)
- Model training during build takes ~5-10 minutes
- Limited storage (~1 GB)
- If inactive for 15+ days, it may spin down

### Deployment Time

First deployment takes ~10-15 minutes because it:

1. Installs all Python packages
2. Trains the model on 10,000 gradient descent iterations
3. Saves the trained model

Subsequent deployments are faster (just pull code from GitHub).

## Troubleshooting

### "Build failed"

- Check the logs in Render dashboard
- Usually due to missing dependencies
- Verify `requirements.txt` is complete

### "Model training timed out"

- Render's free tier has 30-minute build timeout
- Training 10,000 iterations takes ~2-5 minutes (should be fine)
- If it fails, reduce `num_iters` in `train_model.py`

### "Cannot connect to API"

- Give it 1-2 minutes after deployment completes
- Check that the service status shows "Live"
- Try refreshing the browser

### Very slow responses

- Free tier has limited resources
- First request is slower (cold start)
- Normal behavior on free plans

## Making Changes

To update the app:

```bash
# Make changes locally
# ... edit files ...

# Commit and push
git add .
git commit -m "Updated model hyperparameters"
git push origin main
```

Render automatically deploys when you push to `main` branch.

## Optional: Upgrade

If you want better performance:

- Render paid plans start at $7/month
- Get guaranteed uptime, more RAM, faster CPU
- No cold starts

## Next Steps

1. Create GitHub repo
2. Push code to GitHub
3. Connect to Render
4. Share your URL!

Good luck! 🚀
