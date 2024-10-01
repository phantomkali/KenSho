from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode text into BERT embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Taking the mean of the embeddings of the tokenized input sequence
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings
# Example dream interpretations
predicted_interpretations = ["""Transformation and renewal: Snakes shed their skin several times a year, and this process can be seen as a symbol of transformation and renewal. If you are going through a difficult time in your life, this dream could be a sign that you are shedding old patterns, habits, or beliefs that no longer serve you and are making room for new growth and transformation.
Letting go of the past: The snake shedding its skin can also represent the need to let go of the past and leave old versions of yourself behind. If you are holding onto old hurts, resentments, or regrets, this dream could be a sign that it's time to release them and move forward.
Adapting to change: Snakes are highly adaptable creatures, and their ability to shed their skin allows them to adjust to new environments and conditions. This dream could be a reminder that you too are capable of adapting to changes in your life and finding new ways to thrive.
Inner transformation: The snake's shedding skin can also represent an inner transformation or shift in perspective.""",
                             """Dreams about falling from a tall building can have a few different meanings depending on the context of the dream and the emotions you experienced during the dream. Here are some possible interpretations:

Loss of control: Falling from a tall building can represent a feeling of losing control over your life. You may feel like you're plummeting towards the ground with no way to stop yourself, which can be a very unsettling experience.
Fear of failure: The dream could be a reflection of your fear of failure or a fear of not meeting expectations. You may feel like you're falling short of your goals or that you're not living up to the expectations of others.
Overwhelmed: The dream could be a reflection of feeling overwhelmed by the demands of your daily life. You may feel like you're drowning in responsibilities or that you're struggling to keep your head above water.
Lack of support: Falling from a tall building can also represent a lack of support or a feeling of being isolated. You may feel like you're on your own and that there's no one to catch you if you fall.
Emotional turmoil: The dream could be a reflection of emotional turmoil or a feeling of being disconnected from your emotions.""",
                             """Dreaming of being pregnant can have several meanings depending on the context of the dream and the emotions you experienced during the dream. Here are some possible interpretations of your dream:

Fear of responsibility: If you feel overwhelmed by the thought of being pregnant in your dream, it could indicate that you are feeling overwhelmed by responsibilities in your waking life. You may be feeling burdened by the demands of work, family, or other obligations.
Desire for creation: Pregnancy is a process of creation, and dreaming of being pregnant could symbolize a desire for creativity or self-expression in your waking life. You may be longing for the ability to bring something new and beautiful into the world.
Emotional turmoil: The sensation of a heavy body in a pregnancy dream could represent emotional turmoil or a feeling of being weighed down by difficult emotions. You may be experiencing overwhelming emotions or feeling burdened by the demands of daily life.
Personal growth: Pregnancy is a transformative process that involves significant physical, emotional, and psychological changes. Dreaming of being pregnant could symbolize a desire for personal growth or a willingness to undergo significant""",
                             """Dreams about flying are very common and can have different meanings depending on the context of the dream. In this case, there are two aspects of flying that you mentioned in your dream: flying like a bird and changing to your favorite hero and flying.

Flying like a bird:

Birds are often associated with freedom, so flying like a bird in your dream could represent a desire for freedom or a need to break free from constraints in your waking life.
Flying can also symbolize a sense of empowerment and control over one's life. As a bird, you may feel powerful and able to soar to great heights.
Alternatively, flying like a bird could also represent a connection to your inner self or a deeper sense of self-awareness. Birds are known for their intuition and ability to sense changes in their environment, so flying like a bird could symbolize a desire to tap into your own intuition and inner wisdom.
Changing to your favorite hero and flying:

This part of the dream could represent a desire for adventure or a need for excitement in your life. Your favorite hero could represent someone who embodies the qualities you wish to emulate, such as courage, strength, or wisdom.
Flying with your favorite hero could symbolize a desire for companionship or a need for support in your journey towards achieving your goals."""]


actual_interpretations = ["""a snake in your dream shedding its skin could mean you are going through a transformation or releasing an old version of yourself that no longer suits your current timeline. A rat, on the other hand, could reflect that you’re hiding or shunning something.w """,
                         """it could mean that you feel out of control and don’t “have steady footing” in some aspect of your life. On the other end of that spectrum, however, she says that if you find yourself falling in a dream but then start flying, it could represent freedom and trusting the timing of your life.""",
                          """if you have a dream about being pregnant, and you have been working hard on a creative project, the dream may relate to this period of creativity you are experiencing and how you feel about it,” she explains. """,
                          """says that the vivid dream of flying symbolizes freedom and a need for adventure. However if the dream turns into you falling as you fly, it could also mean you feel"""]

# Get embeddings for both texts
for i in range(len(predicted_interpretations)):

    predicted_embedding = get_embeddings(predicted_interpretations[i])
    actual_embedding = get_embeddings(actual_interpretations[i])

# Compute cosine similarity
    cos_sim = cosine_similarity(predicted_embedding.numpy(), actual_embedding.numpy())

    print(f"Cosine Similarity: {cos_sim[0][0]}")
