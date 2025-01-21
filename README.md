#  UIT Data Science Challenge 2024

![image](https://github.com/user-attachments/assets/d372ff64-5f9b-4b6e-acc0-3049ca9b2225)


### Members
| Student Name     | Student ID |
|------------------|------------|
| Nguyen Duy Hoang | 22520467   |
| Ha Huy Hoang     | 22520460   |
| Nguyen Hoang Hiep   | 22520452   |

<video width="320" height="240" controls>
  <source src="./video/UIT-DSC-2024.mp4" type="video/mp4">
</video>

### Project Structure
1. **data:**  [ViMMSD Data](https://www.kaggle.com/datasets/hhhoang/vimmsd-dataset) - Include JSON files and folders containing images corresponding to 3 phases: train, dev (public test), and test (private test).

2. **model:** Contains two notebook files:
   - ocr_text.ipynb - Code for OCR (Optical Character Recognition) to detect text in images
   - main.ipynb - Main code for model training and prediction
3. **approved_post.json, pending_posts.json, predictions.json:** Contains posts and their corresponding predicted values for approved posts, pending posts, and labeled posts.

4. **model.keras**: Stores the model's architecture, weights, and parameters, allowing for model reusability, prediction, and deployment without retraining. 
5. **report**: Contains files related to the report, including .tex, .pdf, and other resources.
6. **results**: Contains the best result file in public test (use mean pooling) and private test (use max pooling).
7. **slide**: Contains the slide presentation for the project.
8. **video**: Contains the video presentation for the project.
9. **deploy_web.py**: Python file containing code to deploy website using Streamlit framework.
10. **requirements.txt**: Store necessary dependencies and requirements.

### How to run a deploy web file

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run deploy_web.py
   ```
