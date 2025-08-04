import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

def predict_label(premise, hypothesis, max_length=None, device=None):
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-medium-mnli")

    # Set maximum length for truncation
    if max_length is None:
        max_length = min(tokenizer.model_max_length, 512)  # Choose a reasonable maximum length

    # Tokenize input text with truncation
    inputs = tokenizer(premise, hypothesis, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # Move inputs to the specified device
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Move model to the specified device
    model.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label = outputs.logits.argmax().item()

    # Get the corresponding label name
    label_name = model.config.id2label[predicted_label]

    return label_name

# Example inputs: 100 pairs of premises and hypotheses
inputs = [
    {"premise": "The quick brown fox jumps over the lazy dog.", "hypothesis": "The lazy dog is jumped by a quick brown fox."},
    {"premise": "A cat is sitting on the windowsill.", "hypothesis": "There is a cat indoors."},
    {"premise": "The sun rises in the east.", "hypothesis": "The moon sets in the west."},
    {"premise": "Eating fruits is good for health.", "hypothesis": "Healthy people eat fruits regularly."},
    {"premise": "The company's profits soared after the new CEO took charge.", "hypothesis": "The shareholders were unhappy with the new CEO's performance."},
    {"premise": "Birds chirp melodiously in the morning.", "hypothesis": "Morning melodies are filled with bird chirps."},
    {"premise": "Education is the key to success.", "hypothesis": "Successful individuals value education as crucial."},
    {"premise": "The aroma of freshly brewed coffee fills the air.", "hypothesis": "A delightful scent emanates from the freshly made coffee."},
    {"premise": "Exercise contributes to a healthy lifestyle.", "hypothesis": "Healthy living involves regular exercise routines."},
    {"premise": "Music soothes the soul.", "hypothesis": "Listening to music brings comfort to one's spirit."},
    {"premise": "The majestic mountains stand tall against the horizon.", "hypothesis": "Mountains create a breathtaking backdrop against the sky."},
    {"premise": "The scent of flowers wafts through the garden.", "hypothesis": "Garden blooms emit a pleasant fragrance."},
    {"premise": "Hard work leads to success.", "hypothesis": "Achievement is a result of diligent effort."},
    {"premise": "Laughter is contagious.", "hypothesis": "The joy of laughter spreads rapidly."},
    {"premise": "Raindrops patter softly on the windowpane.", "hypothesis": "Windows are serenaded by the gentle patter of raindrops."},
    {"premise": "Reading broadens the mind.", "hypothesis": "Intellectual horizons expand through reading."},
    {"premise": "The smell of freshly cut grass fills the air after rainfall.", "hypothesis": "Rainfall brings a refreshing aroma of cut grass."},
    {"premise": "Learning a new language enhances cognitive abilities.", "hypothesis": "Cognitive skills improve with language acquisition."},
    {"premise": "Stars twinkle in the night sky.", "hypothesis": "The night sky sparkles with twinkling stars."},
    {"premise": "The taste of homemade cookies brings back childhood memories.", "hypothesis": "Homemade cookies evoke nostalgic childhood reminiscences."},
    {"premise": "Travel broadens perspectives.", "hypothesis": "Perspectives are enriched through travel experiences."},
    {"premise": "A warm hug can brighten someone's day.", "hypothesis": "Embraces have the power to uplift spirits."},
    {"premise": "The sound of waves crashing against the shore is calming.", "hypothesis": "Calming serenity accompanies the crashing waves on the shore."},
    {"premise": "Creativity flourishes in a nurturing environment.", "hypothesis": "A nurturing environment fosters creative expression."},
    {"premise": "Sunsets paint the sky in hues of orange and pink.", "hypothesis": "The sky is adorned with orange and pink hues during sunsets."},
    {"premise": "Friendship enriches life's journey.", "hypothesis": "Life's journey is enriched by meaningful friendships."},
    {"premise": "The scent of pine fills the forest air.", "hypothesis": "Forest air carries the refreshing fragrance of pine."},
    {"premise": "Patience is a virtue.", "hypothesis": "Virtue is embodied in the quality of patience."},
    {"premise": "The crackling of a fireplace creates a cozy ambiance.", "hypothesis": "Coziness emanates from the crackling fireplace."},
    {"premise": "Learning from mistakes leads to personal growth.", "hypothesis": "Personal growth stems from learning through mistakes."},
    {"premise": "Sunflowers turn their faces towards the sun.", "hypothesis": "Sunflowers seek the sun's warmth as they turn towards it."},
    {"premise": "Humor lightens the mood.", "hypothesis": "A light-hearted mood ensues with the presence of humor."},
    {"premise": "The aroma of freshly baked bread is irresistible.", "hypothesis": "Irresistible scents emanate from freshly baked bread."},
    {"premise": "Kindness is contagious.", "hypothesis": "Acts of kindness spread contagiously."},
    {"premise": "The rustling of leaves creates a soothing melody.", "hypothesis": "Leaves rustle to produce a soothing symphony."},
    {"premise": "Adversity builds resilience.", "hypothesis": "Resilience is forged through facing adversity."},
    {"premise": "The sweetness of ripe strawberries delights the senses.", "hypothesis": "Delightful sensations accompany the taste of ripe strawberries."},
    {"premise": "Artistic expression fosters emotional well-being.", "hypothesis": "Emotional well-being is nurtured through artistic expression."},
    {"premise": "Stars illuminate the night sky.", "hypothesis": "The night sky is illuminated by the twinkling stars."},
    {"premise": "Gratitude cultivates happiness.", "hypothesis": "Happiness flourishes through cultivating gratitude."},
    {"premise": "The fragrance of roses fills the garden.", "hypothesis": "Garden air is permeated with the scent of roses."},
    {"premise": "Courage is the foundation of change.", "hypothesis": "Change is built upon a foundation of courage."},
    {"premise": "The chirping of crickets signals the arrival of evening.", "hypothesis": "Evening arrives with the familiar chirping of crickets."},
    {"premise": "Forgiveness leads to inner peace.", "hypothesis": "Inner peace is attained through the act of forgiveness."},
    {"premise": "The laughter of children is infectious.", "hypothesis": "Infectious laughter fills the air with the joy of children."},
    {"premise": "Rainbows arch across the sky after a rainfall.", "hypothesis": "Rainfall brings about the appearance of colorful rainbows in the sky."},
    {"premise": "Optimism fosters resilience.", "hypothesis": "Resilience is nurtured by an optimistic outlook."},
    {"premise": "The aroma of freshly brewed tea is comforting.", "hypothesis": "Comforting scents arise from freshly brewed tea."},
    {"premise": "Empathy fosters understanding.", "hypothesis": "Understanding is nurtured through empathetic interactions."},
    {"premise": "The sound of thunder echoes through the valley.", "hypothesis": "Valleys resonate with the echoing sound of thunder."},
    {"premise": "Hope fuels perseverance.", "hypothesis": "Perseverance is sustained by a sense of hope."},
    {"premise": "The soft glow of candlelight creates a romantic ambiance.", "hypothesis": "Romantic ambiance is enhanced by the soft glow of candlelight."},
    {"premise": "Acceptance leads to inner peace.", "hypothesis": "Inner peace is achieved through acceptance."},
    {"premise": "The gentle breeze rustles the leaves of trees.", "hypothesis": "Tree leaves sway gently in response to the rustling breeze."},
    {"premise": "Love knows no boundaries.", "hypothesis": "Boundaries dissolve in the face of love."},
    {"premise": "Diligence paves the path to success.", "hypothesis": "Success is attained through the path of diligence."},
    {"premise": "The scent of fresh rain refreshes the earth.", "hypothesis": "Rainfall brings a refreshing scent to the earth."},
    {"premise": "Compassion fosters connection.", "hypothesis": "Connection is deepened through acts of compassion."},
    {"premise": "Whispers of wind rustle through the leaves.", "hypothesis": "Leaves rustle softly with the whispers of the wind."},
    {"premise": "Gratitude is the key to happiness.", "hypothesis": "Happiness is unlocked through the practice of gratitude."},
    {"premise": "The scent of freshly baked cookies evokes fond memories.", "hypothesis": "Fond memories arise with the scent of freshly baked cookies."},
    {"premise": "Perseverance conquers adversity.", "hypothesis": "Adversity is overcome through perseverance."},
    {"premise": "The gentle hum of bees fills the garden.", "hypothesis": "Garden spaces are filled with the gentle hum of bees."},
    {"premise": "Dreams inspire aspirations.", "hypothesis": "Aspirations are ignited by the power of dreams."},
    {"premise": "The scent of pine needles perfumes the forest air.", "hypothesis": "Forest air is perfumed with the aroma of pine needles."},
    {"premise": "Generosity breeds goodwill.", "hypothesis": "Goodwill flourishes in the wake of generosity."},
    {"premise": "The sound of crickets heralds the arrival of dusk.", "hypothesis": "Dusk arrives accompanied by the distinctive sound of crickets."},
    {"premise": "Optimism brightens even the darkest days.", "hypothesis": "Dark days are illuminated by the light of optimism."},
    {"premise": "The aroma of freshly baked bread brings warmth to the kitchen.", "hypothesis": "Kitchen warmth emanates from the scent of freshly baked bread."},
    {"premise": "Friendship fosters camaraderie.", "hypothesis": "Camaraderie flourishes in the presence of friendship."},
    {"premise": "The rustling of leaves creates a tranquil atmosphere.", "hypothesis": "Tranquility envelops the surroundings with the rustling of leaves."},
    {"premise": "Hope springs eternal.", "hypothesis": "Eternal hope resides within the human spirit."},
    {"premise": "The melody of a song carries emotions.", "hypothesis": "Emotions are conveyed through the melodic strains of a song."},
    {"premise": "The scent of rain is carried on the wind.", "hypothesis": "Wind carries the scent of rain through the air."},
    {"premise": "Kindness brings joy to others.", "hypothesis": "Joy is spread through acts of kindness."},
    {"premise": "Stars shimmer in the night sky.", "hypothesis": "Shimmering stars adorn the night sky."},
    {"premise": "The gentle lapping of waves creates a soothing rhythm.", "hypothesis": "A soothing rhythm is created by the gentle lapping of waves."},
    {"premise": "Adversity fosters resilience.", "hypothesis": "Resilience is nurtured through experiences of adversity."},
    {"premise": "The laughter of children brightens the day.", "hypothesis": "Daylight is infused with brightness by the laughter of children."},
    {"premise": "Rainbows paint the sky with vibrant colors.", "hypothesis": "Vibrant colors adorn the sky with the appearance of rainbows."},
    {"premise": "Love conquers all obstacles.", "hypothesis": "Obstacles are overcome by the power of love."},
    {"premise": "The gentle breeze whispers through the trees.", "hypothesis": "Trees sway gently to the whispers of the breeze."},
    {"premise": "Patience leads to understanding.", "hypothesis": "Understanding is attained through the practice of patience."},
    {"premise": "The scent of roses evokes romantic sentiments.", "hypothesis": "Romantic sentiments arise with the scent of roses."},
    {"premise": "Curiosity fuels discovery.", "hypothesis": "Discovery is propelled by the fuel of curiosity."},
    {"premise": "The sound of rainfall soothes the soul.", "hypothesis": "Soothing sensations accompany the sound of rainfall."},
    {"premise": "Kindness fosters compassion.", "hypothesis": "Compassion grows from acts of kindness."},
    {"premise": "The gentle rustle of leaves accompanies a serene evening.", "hypothesis": "Evenings are marked by serenity with the gentle rustle of leaves."},
    {"premise": "Resilience overcomes adversity.", "hypothesis": "Adversity is overcome through resilience."},
    {"premise": "The scent of blooming flowers fills the garden.", "hypothesis": "Garden spaces are perfumed with the scent of blooming flowers."},
    {"premise": "Empathy fosters connection.", "hypothesis": "Connection is deepened through empathetic understanding."},
    {"premise": "The sound of waves crashing against the shore brings tranquility.", "hypothesis": "Tranquility accompanies the crashing waves against the shore."},
    {"premise": "Hope inspires perseverance.", "hypothesis": "Perseverance is fueled by a sense of hope."},
    {"premise": "The scent of freshly brewed coffee awakens the senses.", "hypothesis": "Awakening sensations arise from the scent of freshly brewed coffee."},
    {"premise": "Compassion nurtures empathy.", "hypothesis": "Empathy flourishes through compassionate acts."},
    {"premise": "The warmth of sunlight embraces the earth.", "hypothesis": "Sunlight envelops the earth with its warmth."},
    {"premise": "Optimism breeds resilience.", "hypothesis": "Resilience is cultivated through an optimistic outlook."},
    {"premise": "The chirping of birds signals the arrival of dawn.", "hypothesis": "Dawn arrives accompanied by the melodious chirping of birds."},
    {"premise": "Forgiveness leads to healing.", "hypothesis": "Healing is facilitated by the act of forgiveness."},
    {"premise": "The soft rustle of leaves creates a peaceful ambiance.", "hypothesis": "Peaceful ambiance is created by the soft rustle of leaves."},
    # Add more inputs here
]

# Predict labels for each input using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()
for i in tqdm(range(100), desc='Inference'):
    for input_data in inputs:
        premise = input_data["premise"]
        hypothesis = input_data["hypothesis"]
        predicted_label = predict_label(premise, hypothesis, device=device)
        # print("Premise:", premise)
        # print("Hypothesis:", hypothesis)
        # print("Predicted label:", predicted_label)
        # print()
print()