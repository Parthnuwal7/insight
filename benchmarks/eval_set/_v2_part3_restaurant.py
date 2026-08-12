"""Restaurant reviews for the v2 general eval set. See build_v2_general.py."""


def load(add):
    # --- single_aspect_control (6) ---
    add("The food was absolutely delicious.", "Delicious food",
        "restaurant", "single_aspect_control", "en",
        [("food", "Positive", "food was absolutely delicious")])
    add("Service was painfully slow all evening.", "Slow service",
        "restaurant", "single_aspect_control", "en",
        [("service", "Negative", "Service was painfully slow")])
    add("The ambiance is warm and welcoming.", "Lovely ambiance",
        "restaurant", "single_aspect_control", "en",
        [("ambiance", "Positive", "ambiance is warm and welcoming")])
    add("Portions are far too small for the price.", "Tiny portions",
        "restaurant", "single_aspect_control", "en",
        [("portions", "Negative", "Portions are far too small")])
    add("The staff were friendly and attentive throughout.", "Great staff",
        "restaurant", "single_aspect_control", "en",
        [("staff", "Positive", "staff were friendly and attentive")])
    add("The dessert menu is very limited.", "Limited desserts",
        "restaurant", "single_aspect_control", "en",
        [("dessert menu", "Negative", "dessert menu is very limited")])

    # --- multi_aspect (9) ---
    add("Wonderful food and excellent service. We will be back.", "Great evening",
        "restaurant", "multi_aspect", "en",
        [("food", "Positive", "Wonderful food"),
         ("service", "Positive", "excellent service")])
    add("The pasta was undercooked and the waiter was rude.", "Poor meal",
        "restaurant", "multi_aspect", "en",
        [("pasta", "Negative", "pasta was undercooked"),
         ("waiter", "Negative", "waiter was rude")])
    add("Lovely atmosphere, generous portions, and reasonable prices.", "Good value",
        "restaurant", "multi_aspect", "en",
        [("atmosphere", "Positive", "Lovely atmosphere"),
         ("portions", "Positive", "generous portions"),
         ("prices", "Positive", "reasonable prices")])
    add("The wait time was over an hour and the table was sticky.", "Bad start",
        "restaurant", "multi_aspect", "en",
        [("wait time", "Negative", "wait time was over an hour"),
         ("table", "Negative", "table was sticky")])
    add("Excellent wine list and the steak was cooked perfectly.", "Perfect steak",
        "restaurant", "multi_aspect", "en",
        [("wine list", "Positive", "Excellent wine list"),
         ("steak", "Positive", "steak was cooked perfectly")])
    add("Noisy dining room and the music was far too loud to talk.", "Too noisy",
        "restaurant", "multi_aspect", "en",
        [("dining room", "Negative", "Noisy dining room"),
         ("music", "Negative", "music was far too loud")])
    add("The bread was fresh and the olive oil was excellent quality.", "Great starters",
        "restaurant", "multi_aspect", "en",
        [("bread", "Positive", "bread was fresh"),
         ("olive oil", "Positive", "olive oil was excellent quality")])
    add("Overpriced menu and the parking situation is a nightmare.", "Not convenient",
        "restaurant", "multi_aspect", "en",
        [("menu", "Negative", "Overpriced menu"),
         ("parking", "Negative", "parking situation is a nightmare")])
    add("Fresh ingredients, beautiful presentation, and friendly staff.", "Highly recommend",
        "restaurant", "multi_aspect", "en",
        [("ingredients", "Positive", "Fresh ingredients"),
         ("presentation", "Positive", "beautiful presentation"),
         ("staff", "Positive", "friendly staff")])

    # --- mixed_sentiment (6) ---
    add("The food was outstanding but the service was incredibly slow.", "Great food slow service",
        "restaurant", "mixed_sentiment", "en",
        [("food", "Positive", "food was outstanding"),
         ("service", "Negative", "service was incredibly slow")])
    add("Beautiful decor, shame the food was bland.", "Pretty but bland",
        "restaurant", "mixed_sentiment", "en",
        [("decor", "Positive", "Beautiful decor"),
         ("food", "Negative", "food was bland")])
    add("Staff were lovely but the kitchen got our order wrong twice.", "Nice staff wrong order",
        "restaurant", "mixed_sentiment", "en",
        [("staff", "Positive", "Staff were lovely"),
         ("kitchen", "Negative", "kitchen got our order wrong twice")])
    add("Great value for money, although the seating is uncomfortable.", "Cheap but cramped",
        "restaurant", "mixed_sentiment", "en",
        [("value for money", "Positive", "Great value for money"),
         ("seating", "Negative", "seating is uncomfortable")])
    add("The curry was superb, however the naan arrived cold.", "Superb curry",
        "restaurant", "mixed_sentiment", "en",
        [("curry", "Positive", "curry was superb"),
         ("naan", "Negative", "naan arrived cold")])
    add("Quick service but the coffee tasted burnt.", "Fast but bitter",
        "restaurant", "mixed_sentiment", "en",
        [("service", "Positive", "Quick service"),
         ("coffee", "Negative", "coffee tasted burnt")])

    # --- long_form (5) ---
    add("We booked here for an anniversary dinner and it mostly lived up to "
        "expectations. The ambiance is genuinely special, low lighting and well "
        "spaced tables so you can actually hold a conversation. Our server was "
        "attentive without hovering and knew the menu well. The starters were the "
        "highlight, the scallops in particular were cooked perfectly. The main "
        "course was less impressive, my lamb was noticeably overcooked and dry. "
        "The wine list is extensive but the markup is steep. Dessert made up for "
        "it and the staff brought out a candle without us asking.",
        "Anniversary dinner",
        "restaurant", "long_form", "en",
        [("ambiance", "Positive", "ambiance is genuinely special"),
         ("server", "Positive", "server was attentive"),
         ("starters", "Positive", "starters were the highlight"),
         ("lamb", "Negative", "lamb was noticeably overcooked"),
         ("wine list", "Negative", "markup is steep"),
         ("staff", "Positive", "staff brought out a candle")])
    add("Sadly this place has gone downhill since it changed hands. We used to "
        "come monthly. The menu has been cut in half and the prices have gone up "
        "noticeably. The food quality is not what it was, my risotto was gluey "
        "and clearly reheated. Service was disorganised, we waited twenty minutes "
        "to order and our drinks arrived after the food. The dining room itself "
        "is still lovely and the location is convenient, which is the only reason "
        "I would consider going back.",
        "Not what it was",
        "restaurant", "long_form", "en",
        [("menu", "Negative", "menu has been cut in half"),
         ("prices", "Negative", "prices have gone up"),
         ("food quality", "Negative", "food quality is not what it was"),
         ("service", "Negative", "Service was disorganised"),
         ("dining room", "Positive", "dining room itself is still lovely"),
         ("location", "Positive", "location is convenient")])
    add("Genuinely one of the best meals I have had this year. Everything from "
        "start to finish was considered. The bread arrived warm with excellent "
        "butter. The tasting menu was well paced and each course was distinct. "
        "Portions were judged well, I left satisfied but not uncomfortable. The "
        "staff were knowledgeable and clearly enjoyed working there. Prices are "
        "high but honestly justified given the quality of the ingredients. The "
        "only small criticism is that the restaurant is quite hard to find.",
        "Exceptional meal",
        "restaurant", "long_form", "en",
        [("bread", "Positive", "bread arrived warm"),
         ("tasting menu", "Positive", "tasting menu was well paced"),
         ("portions", "Positive", "Portions were judged well"),
         ("staff", "Positive", "staff were knowledgeable"),
         ("prices", "Negative", "Prices are high"),
         ("ingredients", "Positive", "quality of the ingredients")])
    add("Went for a casual weekday lunch and it was fine, nothing more. The food "
        "was competent, my sandwich was fresh and the soup was properly seasoned. "
        "Service was efficient and we were in and out in forty minutes which "
        "suited us. The interior is a bit tired, the chairs are worn and the "
        "lighting is harsh. Prices are on the higher side for what is essentially "
        "a cafe. It does the job if you need a quick lunch nearby.",
        "Perfectly adequate",
        "restaurant", "long_form", "en",
        [("food", "Positive", "food was competent"),
         ("soup", "Positive", "soup was properly seasoned"),
         ("service", "Positive", "Service was efficient"),
         ("interior", "Negative", "interior is a bit tired"),
         ("lighting", "Negative", "lighting is harsh"),
         ("prices", "Negative", "Prices are on the higher side")])
    add("Booked a table for eight for a birthday and the restaurant handled it "
        "badly. Despite booking three weeks ahead they had us at two separate "
        "tables. The manager was apologetic and did eventually rearrange things, "
        "which I appreciated. Once seated the food was very good, the shared "
        "platters especially were generous and full of flavour. Service after "
        "that point was attentive. But the noise level made conversation across "
        "the table impossible and the bill took half an hour to arrive.",
        "Good food, poor organisation",
        "restaurant", "long_form", "en",
        [("manager", "Positive", "manager was apologetic"),
         ("food", "Positive", "food was very good"),
         ("platters", "Positive", "platters especially were generous"),
         ("service", "Positive", "Service after that point was attentive"),
         ("noise level", "Negative", "noise level made conversation across the table impossible")])

    # --- hindi (2) ---
    # English aspect names, Devanagari evidence -- see _v2_part1_ecommerce.
    add("खाना बहुत स्वादिष्ट था और सेवा भी अच्छी थी।", "बढ़िया खाना",
        "restaurant", "hindi", "hi",
        [("food", "Positive", "खाना बहुत स्वादिष्ट था"),
         ("service", "Positive", "सेवा भी अच्छी थी")])
    add("कीमत बहुत ज़्यादा थी और माहौल शोरगुल वाला था।", "महंगा",
        "restaurant", "hindi", "hi",
        [("price", "Negative", "कीमत बहुत ज़्यादा थी"),
         ("atmosphere", "Negative", "माहौल शोरगुल वाला था")])

    # --- hinglish (2) ---
    add("Khana bahut tasty tha lekin service slow thi.", "Tasty khana",
        "restaurant", "hinglish", "en",
        [("Khana", "Positive", "Khana bahut tasty tha"),
         ("service", "Negative", "service slow thi")])
    add("Ambiance ekdum mast tha aur staff bhi friendly the.", "Mast ambiance",
        "restaurant", "hinglish", "en",
        [("Ambiance", "Positive", "Ambiance ekdum mast tha"),
         ("staff", "Positive", "staff bhi friendly the")])
