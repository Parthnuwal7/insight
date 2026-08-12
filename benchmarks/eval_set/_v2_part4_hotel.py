"""Hotel reviews for the v2 general eval set. See build_v2_general.py."""


def load(add):
    # --- single_aspect_control (6) ---
    add("The room was spotlessly clean.", "Clean room",
        "hotel", "single_aspect_control", "en",
        [("room", "Positive", "room was spotlessly clean")])
    add("The wifi was unusable for the entire stay.", "No wifi",
        "hotel", "single_aspect_control", "en",
        [("wifi", "Negative", "wifi was unusable")])
    add("Breakfast was excellent with plenty of choice.", "Great breakfast",
        "hotel", "single_aspect_control", "en",
        [("breakfast", "Positive", "Breakfast was excellent")])
    add("The bathroom was outdated and poorly maintained.", "Old bathroom",
        "hotel", "single_aspect_control", "en",
        [("bathroom", "Negative", "bathroom was outdated")])
    add("The location is perfect for exploring the city.", "Great location",
        "hotel", "single_aspect_control", "en",
        [("location", "Positive", "location is perfect")])
    add("The beds were extremely uncomfortable.", "Bad beds",
        "hotel", "single_aspect_control", "en",
        [("beds", "Negative", "beds were extremely uncomfortable")])

    # --- multi_aspect (9) ---
    add("Lovely room and the staff were incredibly helpful.", "Great stay",
        "hotel", "multi_aspect", "en",
        [("room", "Positive", "Lovely room"),
         ("staff", "Positive", "staff were incredibly helpful")])
    add("The air conditioning was broken and reception was unresponsive.", "Hot and ignored",
        "hotel", "multi_aspect", "en",
        [("air conditioning", "Negative", "air conditioning was broken"),
         ("reception", "Negative", "reception was unresponsive")])
    add("Comfortable beds, quiet rooms, and an excellent breakfast spread.", "Restful",
        "hotel", "multi_aspect", "en",
        [("beds", "Positive", "Comfortable beds"),
         ("rooms", "Positive", "quiet rooms"),
         ("breakfast", "Positive", "excellent breakfast spread")])
    add("The pool was closed and the gym equipment was broken.", "Facilities closed",
        "hotel", "multi_aspect", "en",
        [("pool", "Negative", "pool was closed"),
         ("gym equipment", "Negative", "gym equipment was broken")])
    add("Check in was fast and the room had a stunning view.", "Smooth arrival",
        "hotel", "multi_aspect", "en",
        [("check in", "Positive", "Check in was fast"),
         ("view", "Positive", "stunning view")])
    add("Thin walls, noisy corridors, and the heating never worked.", "Sleepless",
        "hotel", "multi_aspect", "en",
        [("walls", "Negative", "Thin walls"),
         ("corridors", "Negative", "noisy corridors"),
         ("heating", "Negative", "heating never worked")])
    add("The spa was wonderful and the restaurant served great food.", "Excellent facilities",
        "hotel", "multi_aspect", "en",
        [("spa", "Positive", "spa was wonderful"),
         ("restaurant", "Positive", "restaurant served great food")])
    add("Parking is expensive and the lift was out of service all week.", "Inconvenient",
        "hotel", "multi_aspect", "en",
        [("parking", "Negative", "Parking is expensive"),
         ("lift", "Negative", "lift was out of service")])
    add("Spacious room, modern bathroom, and very friendly reception staff.", "Modern and comfy",
        "hotel", "multi_aspect", "en",
        [("room", "Positive", "Spacious room"),
         ("bathroom", "Positive", "modern bathroom"),
         ("reception staff", "Positive", "friendly reception staff")])

    # --- mixed_sentiment (6) ---
    add("The location is excellent but the rooms are very dated.", "Good spot old rooms",
        "hotel", "mixed_sentiment", "en",
        [("location", "Positive", "location is excellent"),
         ("rooms", "Negative", "rooms are very dated")])
    add("Staff were wonderful, though the breakfast was disappointing.", "Nice staff poor food",
        "hotel", "mixed_sentiment", "en",
        [("staff", "Positive", "Staff were wonderful"),
         ("breakfast", "Negative", "breakfast was disappointing")])
    add("Beautiful lobby, but our room smelled of damp.", "Looks can deceive",
        "hotel", "mixed_sentiment", "en",
        [("lobby", "Positive", "Beautiful lobby"),
         ("room", "Negative", "room smelled of damp")])
    add("Great value for the price, although the wifi kept dropping.", "Cheap but patchy",
        "hotel", "mixed_sentiment", "en",
        [("value", "Positive", "Great value for the price"),
         ("wifi", "Negative", "wifi kept dropping")])
    add("The bed was extremely comfortable but the shower had no pressure.", "Good sleep bad shower",
        "hotel", "mixed_sentiment", "en",
        [("bed", "Positive", "bed was extremely comfortable"),
         ("shower", "Negative", "shower had no pressure")])
    add("The garden was quiet and peaceful, however the check out process was chaotic.",
        "Peaceful but messy exit",
        "hotel", "mixed_sentiment", "en",
        [("garden", "Positive", "garden was quiet and peaceful"),
         ("check out process", "Negative", "check out process was chaotic")])

    # --- long_form (5) ---
    add("Stayed here for four nights on a work trip and it was a solid choice. "
        "The location is the standout feature, five minutes from the station and "
        "walking distance to everything I needed. Check in was efficient even "
        "though I arrived late. The room was compact but very well designed, with "
        "good storage and a genuinely comfortable bed. The bathroom was modern "
        "and the shower had excellent pressure. Breakfast was the weak point, a "
        "limited buffet that ran out of most things by half eight. Wifi was fast "
        "and stable throughout, which matters when you are working.",
        "Solid business stay",
        "hotel", "long_form", "en",
        [("location", "Positive", "location is the standout feature"),
         ("check in", "Positive", "Check in was efficient"),
         ("room", "Positive", "room was compact but very well designed"),
         ("bed", "Positive", "genuinely comfortable bed"),
         ("bathroom", "Positive", "bathroom was modern"),
         ("breakfast", "Negative", "Breakfast was the weak point"),
         ("wifi", "Positive", "Wifi was fast and stable")])
    add("I would not stay here again. The photographs online are clearly several "
        "years old. Our room was tired, the carpet was stained and there was "
        "visible mould around the window frame. I raised it with reception and "
        "they moved us, which was handled politely, but the second room had the "
        "same damp smell. The heating was uncontrollable, either off or "
        "sweltering. Breakfast was actually decent, fresh fruit and good coffee. "
        "The location is convenient but not enough to make up for the state of "
        "the rooms.",
        "Photos are misleading",
        "hotel", "long_form", "en",
        [("room", "Negative", "Our room was tired"),
         ("carpet", "Negative", "carpet was stained"),
         ("reception", "Positive", "handled politely"),
         ("heating", "Negative", "heating was uncontrollable"),
         ("Breakfast", "Positive", "Breakfast was actually decent"),
         ("location", "Positive", "location is convenient")])
    add("A genuinely lovely place for a weekend away. The building has real "
        "character and the staff clearly care about it. Our room overlooked the "
        "garden and was quiet all night. The bed and pillows were excellent. "
        "Breakfast is cooked to order rather than a buffet which made a big "
        "difference to the quality. The spa was small but immaculate and never "
        "crowded. Parking is limited and we had to use a public car park nearby, "
        "which was the only inconvenience. Prices are fair for what you get.",
        "Lovely weekend",
        "hotel", "long_form", "en",
        [("staff", "Positive", "staff clearly care about it"),
         ("room", "Positive", "room overlooked the garden and was quiet"),
         ("bed", "Positive", "bed and pillows were excellent"),
         ("breakfast", "Positive", "Breakfast is cooked to order"),
         ("spa", "Positive", "spa was small but immaculate"),
         ("parking", "Negative", "Parking is limited"),
         ("prices", "Positive", "Prices are fair")])
    add("Booked this for a family holiday and it was a mixed bag. The pool area "
        "is fantastic, clean and well supervised, and the children loved it. "
        "Our family room was spacious enough for four with proper beds rather "
        "than fold outs. However the restaurant was consistently poor, the food "
        "was lukewarm every evening and the choice for children was limited to "
        "chips. Housekeeping was inconsistent, some days thorough and some days "
        "clearly skipped. The reception staff were always friendly and helped us "
        "book excursions.",
        "Great pool, poor restaurant",
        "hotel", "long_form", "en",
        [("pool area", "Positive", "pool area is fantastic"),
         ("family room", "Positive", "family room was spacious"),
         ("restaurant", "Negative", "restaurant was consistently poor"),
         ("housekeeping", "Negative", "Housekeeping was inconsistent"),
         ("reception staff", "Positive", "reception staff were always friendly")])
    add("Arrived to find our booking had not been recorded despite having a "
        "confirmation email. The receptionist was apologetic and found us a room, "
        "but it was a downgrade from what we paid for and no refund was offered "
        "at the time. The room itself was clean and the bed comfortable. Noise "
        "from the street was significant, we could hear traffic all night through "
        "single glazed windows. Breakfast was fine, standard continental. The "
        "hotel did eventually refund the difference after I emailed, so credit to "
        "them for resolving it.",
        "Booking problems",
        "hotel", "long_form", "en",
        [("receptionist", "Positive", "receptionist was apologetic"),
         ("room", "Positive", "room itself was clean"),
         ("bed", "Positive", "bed comfortable"),
         ("noise", "Negative", "Noise from the street was significant"),
         ("breakfast", "Neutral", "Breakfast was fine")])

    # --- hindi (2) ---
    # English aspect names, Devanagari evidence -- see _v2_part1_ecommerce.
    add("कमरा बहुत साफ था और स्टाफ मददगार था।", "अच्छा होटल",
        "hotel", "hindi", "hi",
        [("room", "Positive", "कमरा बहुत साफ था"),
         ("staff", "Positive", "स्टाफ मददगार था")])
    add("बाथरूम गंदा था और नाश्ता बहुत खराब था।", "निराशाजनक",
        "hotel", "hindi", "hi",
        [("bathroom", "Negative", "बाथरूम गंदा था"),
         ("breakfast", "Negative", "नाश्ता बहुत खराब था")])

    # --- hinglish (2) ---
    add("Room ekdum clean tha lekin wifi bilkul kaam nahi kar raha tha.", "Clean room no wifi",
        "hotel", "hinglish", "en",
        [("Room", "Positive", "Room ekdum clean tha"),
         ("wifi", "Negative", "wifi bilkul kaam nahi kar raha tha")])
    add("Location bahut acchi hai aur breakfast bhi tasty tha.", "Acchi location",
        "hotel", "hinglish", "en",
        [("Location", "Positive", "Location bahut acchi hai"),
         ("breakfast", "Positive", "breakfast bhi tasty tha")])
