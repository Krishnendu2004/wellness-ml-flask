from flask import Flask, request, render_template, jsonify
import os
import io
import base64
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "food_mobilenet_model.h5"))
CLASS_NAMES_TXT = os.getenv(
    "CLASS_NAMES_PATH", os.path.join(BASE_DIR, "class_names.txt")
)
CALORIES_CSV = os.getenv(
    "CALORIES_CSV_PATH", os.path.join(BASE_DIR, "calories_lookup.csv")
)

model = None

with open(CLASS_NAMES_TXT, "r") as f:
    class_names = [line.strip() for line in f]

df = pd.read_csv(CALORIES_CSV)
df.columns = df.columns.str.strip()
calories_lookup = df.set_index("Food")["Calories"].to_dict()

print("MODEL exists:", os.path.exists(MODEL_PATH))
print("CLASS file exists:", os.path.exists(CLASS_NAMES_TXT))
print("CSV exists:", os.path.exists(CALORIES_CSV))


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(image, target_size=(256, 256)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4MB request cap


@app.before_request
def require_api_key_for_api_routes():
    if request.path.startswith("/api/"):
        expected = os.getenv("ML_API_KEY_EXPECTED")
        provided = request.headers.get("X-ML-API-Key")
        if not expected or provided != expected:
            return jsonify({"error": "Unauthorized"}), 401


@app.errorhandler(413)
def request_entity_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Payload too large"}), 413
    return "Payload too large", 413


def _read_input(name, cast=None, default=None):
    value = None
    if request.method in ("POST", "PUT", "PATCH") and request.is_json:
        payload = request.get_json(silent=True) or {}
        value = payload.get(name)
    if value is None:
        value = request.values.get(name, default)
    if value is None or value == "":
        return default
    if cast is not None:
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default
    return value


def _analyze_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi_value = round(weight_kg / (height_m * height_m), 2)
    min_normal_weight = 18.5 * (height_m * height_m)
    max_normal_weight = 24.9 * (height_m * height_m)

    if bmi_value < 18.5:
        return {
            "bmi": bmi_value,
            "category": "Underweight",
            "distance": round(min_normal_weight - weight_kg, 2),
            "risk": "Moderate Health Risk",
            "fitness": "Strength training + calorie surplus diet",
            "insight": "Low body mass may affect stamina and strength.",
            "advice": "Increase calorie intake with protein-rich foods.",
        }
    if bmi_value < 25:
        return {
            "bmi": bmi_value,
            "category": "Normal",
            "distance": 0.0,
            "risk": "Low Health Risk",
            "fitness": "Maintain routine with balanced training",
            "insight": "Healthy body composition.",
            "advice": "Maintain your diet and training routine.",
        }
    if bmi_value < 30:
        return {
            "bmi": bmi_value,
            "category": "Overweight",
            "distance": round(max_normal_weight - weight_kg, 2),
            "risk": "Moderate Health Risk",
            "fitness": "Cardio + calorie deficit with protein intake",
            "insight": "Extra weight may impact agility and endurance.",
            "advice": "Focus on cardio and balanced nutrition.",
        }
    return {
        "bmi": bmi_value,
        "category": "Obese",
        "distance": round(max_normal_weight - weight_kg, 2),
        "risk": "High Health Risk",
        "fitness": "Low-impact cardio + supervised fat-loss program",
        "insight": "High risk of fitness and metabolic issues.",
        "advice": "Consult a professional and prioritize fat loss.",
    }


def _sleep_range_for_age(age):
    if age <= 13:
        return 9, 11
    if age <= 17:
        return 8, 10
    if age <= 64:
        return 7, 9
    return 7, 8

@app.route("/ping")
def ping():
    return "SERVER WORKING"



# ---------------- Home Page ----------------
@app.route("/")
def home():
    return render_template("splash.html")

# ---------------- Dashboard ----------------
@app.route("/dashboard")
def dashboard():
   return render_template("dashboard.html")


  


# ---------------- BMI Calculator ----------------
@app.route("/bmi")
def bmi():
    height = request.args.get("height", type=float)
    weight = request.args.get("weight", type=float)

    bmi_value = None
    category = None
    insight = None
    advice = None
    distance = None
    risk = None
    fitness = None

    if height and weight:
        analysis = _analyze_bmi(height, weight)
        bmi_value = analysis["bmi"]
        category = analysis["category"]
        insight = analysis["insight"]
        advice = analysis["advice"]
        distance = analysis["distance"]
        risk = analysis["risk"]
        fitness = analysis["fitness"]

    return render_template(
        "bmi.html",
        bmi_value=bmi_value,
        category=category,
        insight=insight,
        advice=advice,
        distance=distance,
        risk=risk,
        fitness=fitness
    )

@app.route("/calorie", methods=["GET"])
def calorie():

    # ---------- BODY CALORIE ----------
    age = request.args.get("age", type=int)
    gender = request.args.get("gender")
    height = request.args.get("height", type=float)
    weight = request.args.get("weight", type=float)

    body_calories = None

    if age and gender and height and weight:
        if gender == "male":
            body_calories = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            body_calories = 10 * weight + 6.25 * height - 5 * age - 161

        body_calories = round(body_calories, 2)

    # ---------- FOOD CALORIE ----------
    food = request.args.get("food")
    quantity = request.args.get("quantity", type=float)

    food_calories = None
    if food and quantity:
     food_calories = round(quantity, 2)
    food_db = {
        "rice": 130,
        "chapati": 120,
        "egg": 155,
        "chicken": 165,
        "banana": 89,
        "milk": 42,
        "oats": 389,
        "burger": 295,
        "pizza": 266,
        "french_fries": 312,
        "fried_chicken": 246,
        "shawarma": 290,
        "noodles": 138,
        "samosa": 262,
        "chocolate": 546,
        "soft_drink": 42,
        "brown_rice": 123,
        "quinoa": 120,
        "whole_wheat_pasta": 124,
        "sweet_potato": 86,
        "avocado": 160,
        "peanut_butter": 588,
        "almonds": 579,
        "walnuts": 654,
        "whey_protein": 412,
        "egg_white": 52,
        "grilled_chicken": 165,
        "turkey": 135,
        "salmon": 208,
        "tuna": 132,
        "paneer": 265,
        "tofu": 76,
        "soy_chunks": 345,
        "greek_yogurt": 59,
        "pasta": 131,
        "lasagna": 135,
        "ramen": 436,
        "sushi": 130,
        "tacos": 226,
        "burrito": 206,
        "paella": 158,
        "fried_rice": 163,
        "spinach": 23,
        "broccoli": 34,
        "carrot": 41,
        "beetroot": 43,
        "capsicum": 31,
        "zucchini": 17,
        "mushroom": 22,
        "apple": 52,
        "orange": 47,
        "mango": 60,
        "pineapple": 50,
        "berries": 57,
        "dates": 277,
        "raisins": 299,
        "coconut_water": 19,
        "lemon_water": 22,
        "sports_drink": 60,
        "protein_shake": 120,
        "smoothie": 80,
        "black_coffee": 2
    }

    if food and quantity and food in food_db:
        food_calories = round((food_db[food] * quantity) / 100, 2)

    # ✅ ONE RETURN AT THE END
    return render_template(
        "calorie.html",
        body_calories=body_calories,
        food_calories=food_calories
    )


# ---------------- Water Intake ----------------
@app.route("/water")
def water():
    # ----- Existing daily water intake -----
    weight = request.args.get("weight")
    water_needed = None

    # ----- New hydration loss inputs -----
    duration = request.args.get("duration")
    weather = request.args.get("weather")

    sweat_loss = None
    extra_water = None
    risk = None

    # Existing logic (UNCHANGED)
    if weight:
        weight = float(weight)
        water_needed = round(weight * 0.033, 2)

    # New hydration loss logic
    if duration and weather:
        duration = float(duration)

        sweat_rate = 0.5  # liters/hour (normal)
        if weather == "hot":
            sweat_rate += 0.3

        sweat_loss = round(sweat_rate * duration, 2)
        extra_water = sweat_loss

        if sweat_loss > 2:
            risk = "High Dehydration Risk"
        elif sweat_loss > 1:
            risk = "Moderate Dehydration Risk"
        else:
            risk = "Low Dehydration Risk"

    return render_template(
        "water.html",
        water_needed=water_needed,
        sweat_loss=sweat_loss,
        extra_water=extra_water,
        risk=risk
    )

# ---------------- Ideal Weight ----------------
@app.route("/ideal-weight")
def ideal_weight():
    height = request.args.get("height")
    gender = request.args.get("gender")

    ideal = None

    if height and gender:
        height = float(height)

        if gender == "male":
            ideal = 50 + 0.9 * (height - 152)
        else:
            ideal = 45.5 + 0.9 * (height - 152)

        ideal = round(ideal, 2)

    return render_template("ideal_weight.html", ideal=ideal)
@app.route("/sleep")
def sleep():
    age = request.args.get("age")
    hours = request.args.get("hours")

    recommendation = None
    status = None
    warning = None

    if age and hours:
        age = int(age)
        hours = float(hours)

        # Recommended sleep by age
        if age <= 13:
            min_sleep, max_sleep = 9, 11
        elif age <= 17:
            min_sleep, max_sleep = 8, 10
        elif age <= 64:
            min_sleep, max_sleep = 7, 9
        else:
            min_sleep, max_sleep = 7, 8

        # Analysis
        if hours < min_sleep:
            status = "Poor Sleep"
            warning = "High stress risk, low energy, reduced focus"
            recommendation = "Try sleeping earlier, reduce screen time, and maintain a fixed sleep schedule."
        elif hours > max_sleep:
            status = "Oversleeping"
            warning = "May cause fatigue or low motivation"
            recommendation = "Maintain a consistent wake-up time and stay active during the day."
        else:
            status = "Healthy Sleep"
            warning = "Stress levels are normal"
            recommendation = "Great job! Maintain your current sleep routine."

    return render_template(
        "sleep.html",
        status=status,
        warning=warning,
        recommendation=recommendation
    )
@app.route("/recovery")
def recovery():
    training_hours = request.args.get("training")
    sleep_hours = request.args.get("sleep")
    soreness = request.args.get("soreness")

    status = None
    risk = None
    recommendation = None
    fatigue = None
    recovery_score = None
    focus = None

    if training_hours and sleep_hours and soreness:
        training_hours = float(training_hours)
        sleep_hours = float(sleep_hours)

        score = 100  # start with perfect recovery

        # ---- Training load impact ----
        if training_hours > 3:
            score -= 30
        elif training_hours > 2:
            score -= 20
        else:
            score -= 10

        # ---- Sleep impact ----
        if sleep_hours < 6:
            score -= 30
        elif sleep_hours < 7:
            score -= 20
        else:
            score -= 10

        # ---- Muscle soreness impact ----
        if soreness == "yes":
            score -= 20

        # ---- Clamp score ----
        score = max(0, min(100, score))
        recovery_score = score

        # ---- Interpret score ----
        if score >= 75:
            status = "Well Recovered"
            fatigue = "Low"
            risk = "Low Injury Risk"
            recommendation = "Maintain routine. Focus on hydration and mobility."
            focus = "Performance Maintenance"

        elif score >= 50:
            status = "Moderate Recovery"
            fatigue = "Moderate"
            risk = "Medium Injury Risk"
            recommendation = "Add stretching, light recovery session, and improve sleep."
            focus = "Active Recovery"

        else:
            status = "Poor Recovery"
            fatigue = "High"
            risk = "High Injury Risk"
            recommendation = "Reduce training load, prioritize sleep, nutrition, and rest."
            focus = "Rest & Rehabilitation"

    return render_template(
        "recovery.html",
        status=status,
        fatigue=fatigue,
        risk=risk,
        recovery_score=recovery_score,
        focus=focus,
        recommendation=recommendation
    )
# Route to display the image calorie predictor page
@app.route("/image-calorie", methods=["GET"])
def image_calorie_page():
    # Just render the page with the form
    return render_template("image_calorie.html", prediction=None)

@app.route("/predict_image", methods=["POST"])
def predict_image():

    if "food_image" not in request.files:
        return render_template("image_calorie.html", error="No image uploaded")

    img_file = request.files["food_image"]
    if img_file.filename == "":
        return render_template("image_calorie.html", error="No image selected")

    # Preprocess and predict
    img_array = preprocess_image(img_file)
    preds = get_model().predict(img_array)

    top_idx = int(np.argmax(preds[0]))
    food_name = class_names[top_idx]
    calories = calories_lookup.get(food_name, "Not available")

    # ---------- CALORIE CATEGORY ----------
    category = None
    suggestion = None

    if isinstance(calories, (int, float)):
        if calories <= 100:
            category = "Low Calorie"
            suggestion = "Good for weight loss or light snacks."
        elif calories <= 300:
            category = "Moderate Calorie"
            suggestion = "Balanced option. Maintain portion control."
        elif calories <= 500:
            category = "High Calorie"
            suggestion = "Reduce portion or add physical activity."
        else:
            category = "Very High Calorie"
            suggestion = "Consume occasionally. Best post-workout."
    else:
        category = "Unknown"
        suggestion = "Calorie data not available."

    # Return to template
    return render_template(
        "image_calorie.html",
        prediction={
            "food": food_name,
            "calories": calories,
            "category": category,
            "suggestion": suggestion
        }
    )



@app.route("/match-fitness")
def match_fitness():
    sleep = request.args.get("sleep")
    training = request.args.get("training")
    soreness = request.args.get("soreness")

    status = None
    readiness = None
    recommendation = None

    if sleep and training and soreness:
        sleep = float(sleep)

        if sleep >= 7 and training != "heavy" and soreness == "no":
            status = "Match Ready"
            readiness = "High"
            recommendation = "You are fit to play. Maintain warm-up and hydration."
        elif sleep >= 6:
            status = "Partially Ready"
            readiness = "Moderate"
            recommendation = "Light training recommended. Avoid full match load."
        else:
            status = "Not Match Ready"
            readiness = "Low"
            recommendation = "Rest, recovery, and proper sleep needed before playing."

    return render_template(
        "match_fitness.html",
        status=status,
        readiness=readiness,
        recommendation=recommendation
    )
@app.route("/training_load")
def training_load():
    intensity = request.args.get("intensity")
    duration = request.args.get("duration")
    sleep = request.args.get("sleep")

    fatigue = None
    advice = None

    if intensity and duration and sleep:
        duration = int(duration)
        sleep = float(sleep)

        score = 0

        if intensity == "high":
            score += 3
        elif intensity == "medium":
            score += 2
        else:
            score += 1

        if duration > 90:
            score += 3
        elif duration > 60:
            score += 2
        else:
            score += 1

        if sleep < 6:
            score += 3
        elif sleep < 7:
            score += 2
        else:
            score += 1

        if score <= 4:
            fatigue = "Low"
            advice = "You are fit. Normal training recommended."
        elif score <= 7:
            fatigue = "Moderate"
            advice = "Monitor fatigue. Light recovery suggested."
        else:
            fatigue = "High"
            advice = "High injury risk. Rest or recovery session advised."

    return render_template(
        "training_load.html",
        fatigue=fatigue,
        advice=advice
    )
@app.route("/diet")
def diet():
    age = request.args.get("age", type=int)
    gender = request.args.get("gender")
    height = request.args.get("height", type=float)
    weight = request.args.get("weight", type=float)
    intensity = request.args.get("intensity")
    goal = request.args.get("goal")
    day = request.args.get("day")
    position = request.args.get("position")

    calories = None
    macros = {}


    # ✅ DEFINE FOODS FIRST (IMPORTANT)
    foods = {
        "protein": [
            "Eggs", "Chicken breast", "Fish",
            "Paneer", "Greek yogurt", "Lentils (Dal)"
        ],
        "carbs": [
            "Rice", "Chapati", "Oats",
            "Potatoes", "Banana", "Pasta"
        ],
        "fats": [
            "Nuts", "Seeds", "Olive oil",
            "Peanut butter", "Avocado"
        ]
    }


            # ----- Meal Timing Recommendations -----
    meal_timing = {}

    if day == "match":
            meal_timing["pre"] = "High carbs + moderate protein (Rice, pasta, banana, oats)"
            meal_timing["post"] = "Fast protein + carbs (Protein shake, eggs, fruits)"
            meal_timing["recovery"] = "Balanced meal with protein & healthy fats"

    elif intensity == "high":
            meal_timing["pre"] = "Carb-rich meal 2–3 hrs before training"
            meal_timing["post"] = "Protein + carbs within 30 minutes after training"
            meal_timing["recovery"] = "Light dinner with protein & vegetables"

    elif intensity == "medium":
            meal_timing["pre"] = "Balanced carbs and protein"
            meal_timing["post"] = "Protein-rich foods (eggs, paneer, dal)"
            meal_timing["recovery"] = "Normal balanced meal"

    else:
            meal_timing["pre"] = "Light meal (fruits, yogurt)"
            meal_timing["post"] = "Normal protein intake"
            meal_timing["recovery"] = "Low-carb, high-protein meal"

        # ----- Weekly Football Meal Plan -----
    weekly_plan = {}

    if position == "fwd":
            weekly_plan = {
                "Monday": "Oats + banana | Rice, chicken | Pasta + veggies",
                "Tuesday": "Eggs + toast | Rice, fish | Chapati + dal",
                "Wednesday": "Smoothie | Chicken wrap | Sweet potato + paneer",
                "Thursday": "Oats + nuts | Rice, chicken | Pasta",
                "Friday": "Banana + yogurt | Light carbs | Early dinner",
                "Match Day": "Oats + honey | Rice + chicken | Recovery shake",
                "Sunday": "Rest day – balanced meals"
            }

    elif position == "mid":
            weekly_plan = {
                "Monday": "Oats + fruits | Rice, fish | Chapati + dal",
                "Tuesday": "Eggs + toast | Chicken rice | Veg pasta",
                "Wednesday": "Smoothie | Paneer bowl | Sweet potato",
                "Thursday": "Oats | Fish curry + rice | Light dinner",
                "Friday": "Fruits + yogurt | Rice | Early dinner",
                "Match Day": "Carb loading | Light lunch | Recovery meal",
                "Sunday": "Rest + hydration"
            }

    else:  # GK or Defender
            weekly_plan = {
                "Monday": "Eggs + toast | Rice, chicken | Veg curry",
                "Tuesday": "Oats | Fish + rice | Paneer",
                "Wednesday": "Smoothie | Chicken wrap | Soup",
                "Thursday": "Eggs | Rice + dal | Veggies",
                "Friday": "Light carbs | Protein meal | Early sleep",
                "Match Day": "Balanced carbs | Light lunch | Recovery food",
                "Sunday": "Rest day meals"
            }

    if age and gender and height and weight and intensity and goal and day and position:

        # ----- BMR -----
        if gender == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        calories = bmr

        # ----- Training intensity -----
        if intensity == "low":
            calories += 300
        elif intensity == "medium":
            calories += 500
        elif intensity == "high":
            calories += 700

        # ----- Goal adjustment -----
        if goal == "gain":
            calories += 300
        elif goal == "loss":
            calories -= 300

        # ----- Day type adjustment -----
        if day == "match":
            calories *= 1.15
        elif day == "rest":
            calories *= 0.80

        calories = round(calories)

        # ----- Base macro split -----
        protein_pct = 0.25
        carb_pct = 0.55
        fat_pct = 0.20

        # ----- Position-based macro adjustment -----
        if position == "gk":
            protein_pct += 0.05
            carb_pct -= 0.05

        elif position == "mid":
            carb_pct += 0.05
            protein_pct -= 0.05

        elif position == "fwd":
            carb_pct += 0.03
            protein_pct += 0.02
            fat_pct -= 0.05

        # ----- Normalize -----
        total = protein_pct + carb_pct + fat_pct
        protein_pct /= total
        carb_pct /= total
        fat_pct /= total

        macros = {
            "protein": round(calories * protein_pct / 4),
            "carbs": round(calories * carb_pct / 4),
            "fats": round(calories * fat_pct / 9)
        }

        


    return render_template(
        "diet.html",
        calories=calories,
        macros=macros,
        foods=foods,
        meal_timing=meal_timing,
        weekly_plan=weekly_plan
    )
@app.route("/api/bmi", methods=["GET", "POST"])
def api_bmi():
    height = _read_input("height", float)
    weight = _read_input("weight", float)
    if not height or not weight:
        return jsonify({"error": "height and weight are required"}), 400
    return jsonify(_analyze_bmi(height, weight))


@app.route("/api/calorie", methods=["GET", "POST"])
def api_calorie():
    age = _read_input("age", int)
    gender = _read_input("gender")
    height = _read_input("height", float)
    weight = _read_input("weight", float)
    food = _read_input("food")
    quantity = _read_input("quantity", float)

    daily_calories = None
    if age and gender and height and weight:
        if str(gender).lower() == "male":
            daily_calories = round(10 * weight + 6.25 * height - 5 * age + 5, 2)
        else:
            daily_calories = round(10 * weight + 6.25 * height - 5 * age - 161, 2)

    food_db = {
        "rice": 130,
        "chapati": 120,
        "egg": 155,
        "chicken": 165,
        "banana": 89,
        "milk": 42,
        "oats": 389,
        "burger": 295,
        "pizza": 266,
        "french_fries": 312,
        "fried_chicken": 246,
        "shawarma": 290,
        "noodles": 138,
        "samosa": 262,
        "chocolate": 546,
        "soft_drink": 42,
        "brown_rice": 123,
        "quinoa": 120,
        "whole_wheat_pasta": 124,
        "sweet_potato": 86,
        "avocado": 160,
        "peanut_butter": 588,
        "almonds": 579,
        "walnuts": 654,
        "whey_protein": 412,
        "egg_white": 52,
        "grilled_chicken": 165,
        "turkey": 135,
        "salmon": 208,
        "tuna": 132,
        "paneer": 265,
        "tofu": 76,
        "soy_chunks": 345,
        "greek_yogurt": 59,
        "pasta": 131,
        "lasagna": 135,
        "ramen": 436,
        "sushi": 130,
        "tacos": 226,
        "burrito": 206,
        "paella": 158,
        "fried_rice": 163,
        "spinach": 23,
        "broccoli": 34,
        "carrot": 41,
        "beetroot": 43,
        "capsicum": 31,
        "zucchini": 17,
        "mushroom": 22,
        "apple": 52,
        "orange": 47,
        "mango": 60,
        "pineapple": 50,
        "berries": 57,
        "dates": 277,
        "raisins": 299,
        "coconut_water": 19,
        "lemon_water": 22,
        "sports_drink": 60,
        "protein_shake": 120,
        "smoothie": 80,
        "black_coffee": 2,
    }

    food_calories = None
    if food and quantity and food in food_db:
        food_calories = round((food_db[food] * quantity) / 100, 2)

    return jsonify(
        {
            "daily_calories": daily_calories,
            "food_calories": food_calories,
        }
    )


@app.route("/api/water", methods=["GET", "POST"])
def api_water():
    weight = _read_input("weight", float)
    duration = _read_input("duration", float)
    weather = _read_input("weather")

    water_intake_liters = round(weight * 0.033, 2) if weight else None
    sweat_loss = None
    dehydration_risk = None

    if duration and weather:
        sweat_rate = 0.5
        if str(weather).lower() == "hot":
            sweat_rate += 0.3
        sweat_loss = round(sweat_rate * duration, 2)
        if sweat_loss > 2:
            dehydration_risk = "High Dehydration Risk"
        elif sweat_loss > 1:
            dehydration_risk = "Moderate Dehydration Risk"
        else:
            dehydration_risk = "Low Dehydration Risk"

    return jsonify(
        {
            "water_intake_liters": water_intake_liters,
            "sweat_loss_liters": sweat_loss,
            "dehydration_risk": dehydration_risk,
        }
    )


@app.route("/api/ideal_weight", methods=["GET", "POST"])
def api_ideal_weight():
    height = _read_input("height", float)
    gender = _read_input("gender")
    if not height or not gender:
        return jsonify({"error": "height and gender are required"}), 400
    if str(gender).lower() == "male":
        ideal = 50 + 0.9 * (height - 152)
    else:
        ideal = 45.5 + 0.9 * (height - 152)
    return jsonify({"ideal_weight_kg": round(ideal, 2)})


@app.route("/api/sleep", methods=["GET", "POST"])
def api_sleep():
    age = _read_input("age", int)
    hours = _read_input("hours", float)
    if age is None:
        return jsonify({"error": "age is required"}), 400

    min_sleep, max_sleep = _sleep_range_for_age(age)
    recommended_sleep_hours = f"{min_sleep}-{max_sleep}"
    status = None

    if hours is not None:
        if hours < min_sleep:
            status = "Poor Sleep"
        elif hours > max_sleep:
            status = "Oversleeping"
        else:
            status = "Healthy Sleep"

    return jsonify(
        {
            "recommended_sleep_hours": recommended_sleep_hours,
            "sleep_status": status,
            "hours_logged": hours,
        }
    )


@app.route("/api/recovery", methods=["GET", "POST"])
def api_recovery():
    training_hours = _read_input("training", float)
    sleep_hours = _read_input("sleep", float)
    soreness = _read_input("soreness")
    if training_hours is None or sleep_hours is None or soreness is None:
        return jsonify({"error": "training, sleep, and soreness are required"}), 400

    score = 100
    if training_hours > 3:
        score -= 30
    elif training_hours > 2:
        score -= 20
    else:
        score -= 10

    if sleep_hours < 6:
        score -= 30
    elif sleep_hours < 7:
        score -= 20
    else:
        score -= 10

    if str(soreness).lower() == "yes":
        score -= 20

    score = max(0, min(100, score))
    return jsonify({"recovery_score": score})


@app.route("/api/match_fitness", methods=["GET", "POST"])
def api_match_fitness():
    sleep = _read_input("sleep", float)
    training = _read_input("training")
    soreness = _read_input("soreness")
    if sleep is None or training is None or soreness is None:
        return jsonify({"error": "sleep, training, and soreness are required"}), 400

    if sleep >= 7 and str(training).lower() != "heavy" and str(soreness).lower() == "no":
        match_fitness_score = 90
        fitness_level = "High"
    elif sleep >= 6:
        match_fitness_score = 70
        fitness_level = "Moderate"
    else:
        match_fitness_score = 40
        fitness_level = "Low"

    return jsonify(
        {
            "match_fitness_score": match_fitness_score,
            "fitness_level": fitness_level,
        }
    )


@app.route("/api/training_load", methods=["GET", "POST"])
def api_training_load():
    intensity = _read_input("intensity")
    duration = _read_input("duration", int)
    sleep = _read_input("sleep", float)
    if intensity is None or duration is None or sleep is None:
        return jsonify({"error": "intensity, duration, and sleep are required"}), 400

    score = 0
    if str(intensity).lower() == "high":
        score += 3
    elif str(intensity).lower() == "medium":
        score += 2
    else:
        score += 1

    if duration > 90:
        score += 3
    elif duration > 60:
        score += 2
    else:
        score += 1

    if sleep < 6:
        score += 3
    elif sleep < 7:
        score += 2
    else:
        score += 1

    if score <= 4:
        training_load = "Low"
        recommendation = "You are fit. Normal training recommended."
    elif score <= 7:
        training_load = "Moderate"
        recommendation = "Monitor fatigue. Light recovery suggested."
    else:
        training_load = "High"
        recommendation = "High injury risk. Rest or recovery session advised."

    return jsonify(
        {
            "training_load": training_load,
            "recommendation": recommendation,
            "score": score,
        }
    )


@app.route("/api/diet", methods=["GET", "POST"])
def api_diet():
    age = _read_input("age", int)
    gender = _read_input("gender")
    height = _read_input("height", float)
    weight = _read_input("weight", float)
    intensity = _read_input("intensity")
    goal = _read_input("goal")
    day = _read_input("day")
    position = _read_input("position")

    required = [age, gender, height, weight, intensity, goal, day, position]
    if any(v is None for v in required):
        return (
            jsonify(
                {
                    "error": "age, gender, height, weight, intensity, goal, day, and position are required"
                }
            ),
            400,
        )

    if str(gender).lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    calories = bmr
    if intensity == "low":
        calories += 300
    elif intensity == "medium":
        calories += 500
    elif intensity == "high":
        calories += 700

    if goal == "gain":
        calories += 300
    elif goal == "loss":
        calories -= 300

    if day == "match":
        calories *= 1.15
    elif day == "rest":
        calories *= 0.80

    daily_calories = round(calories)

    protein_pct, carb_pct, fat_pct = 0.25, 0.55, 0.20
    if position == "gk":
        protein_pct += 0.05
        carb_pct -= 0.05
    elif position == "mid":
        carb_pct += 0.05
        protein_pct -= 0.05
    elif position == "fwd":
        carb_pct += 0.03
        protein_pct += 0.02
        fat_pct -= 0.05

    total = protein_pct + carb_pct + fat_pct
    protein_pct /= total
    carb_pct /= total
    fat_pct /= total

    macros = {
        "protein": round(daily_calories * protein_pct / 4),
        "carbs": round(daily_calories * carb_pct / 4),
        "fats": round(daily_calories * fat_pct / 9),
    }

    return jsonify({"daily_calories": daily_calories, "macros": macros})


@app.route("/api/predict_image_json", methods=["POST"])
def api_predict_image_json():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_base64") or payload.get("image_b64")
    if not image_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        if "," in image_b64 and image_b64.split(",", 1)[0].startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        if len(image_b64) > 5_500_000:
            return jsonify({"error": "image payload too large"}), 413
        image_bytes = base64.b64decode(image_b64, validate=True)
        image_stream = io.BytesIO(image_bytes)
        img_array = preprocess_image(image_stream)
        preds = get_model().predict(img_array)
        top_idx = int(np.argmax(preds[0]))
        food_name = class_names[top_idx]
        calories = calories_lookup.get(food_name, "Not available")
    except Exception as exc:
        return jsonify({"error": f"Invalid image payload: {exc}"}), 400

    category = None
    suggestion = None
    if isinstance(calories, (int, float)):
        if calories <= 100:
            category = "Low Calorie"
            suggestion = "Good for weight loss or light snacks."
        elif calories <= 300:
            category = "Moderate Calorie"
            suggestion = "Balanced option. Maintain portion control."
        elif calories <= 500:
            category = "High Calorie"
            suggestion = "Reduce portion or add physical activity."
        else:
            category = "Very High Calorie"
            suggestion = "Consume occasionally. Best post-workout."
    else:
        category = "Unknown"
        suggestion = "Calorie data not available."

    return jsonify(
        {
            "food": food_name,
            "calories": calories,
            "category": category,
            "suggestion": suggestion,
        }
    )


if __name__ == "__main__":
    app.run(debug=False)

