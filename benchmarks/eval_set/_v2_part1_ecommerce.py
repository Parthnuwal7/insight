"""E-commerce reviews for the v2 general eval set. See build_v2_general.py."""


def load(add):
    # =====================================================================
    # E-COMMERCE (30)
    # =====================================================================

    # --- single_aspect_control (6) ---
    add("The delivery was incredibly fast, arrived in two days.", "Fast delivery",
        "ecommerce", "single_aspect_control", "en",
        [("delivery", "Positive", "delivery was incredibly fast")])
    add("Packaging was terrible, the box was completely crushed.", "Bad packaging",
        "ecommerce", "single_aspect_control", "en",
        [("packaging", "Negative", "Packaging was terrible")])
    add("The price is very reasonable for what you get.", "Good price",
        "ecommerce", "single_aspect_control", "en",
        [("price", "Positive", "price is very reasonable")])
    add("Customer service never responded to my emails.", "No response",
        "ecommerce", "single_aspect_control", "en",
        [("customer service", "Negative", "Customer service never responded")])
    add("The refund was processed without any hassle.", "Easy refund",
        "ecommerce", "single_aspect_control", "en",
        [("refund", "Positive", "refund was processed without any hassle")])
    add("Product quality is disappointing for this price point.", "Poor quality",
        "ecommerce", "single_aspect_control", "en",
        [("quality", "Negative", "quality is disappointing")])

    # --- multi_aspect (9) ---
    add("Great product and the shipping was quick too. Very satisfied.", "Great buy",
        "ecommerce", "multi_aspect", "en",
        [("product", "Positive", "Great product"),
         ("shipping", "Positive", "shipping was quick")])
    add("The item arrived damaged and the seller refused a replacement.", "Damaged item",
        "ecommerce", "multi_aspect", "en",
        [("item", "Negative", "item arrived damaged"),
         ("seller", "Negative", "seller refused a replacement")])
    add("Excellent packaging, fast delivery, and the product works perfectly.", "All good",
        "ecommerce", "multi_aspect", "en",
        [("packaging", "Positive", "Excellent packaging"),
         ("delivery", "Positive", "fast delivery"),
         ("product", "Positive", "product works perfectly")])
    add("Poor build quality and the price is far too high.", "Not worth it",
        "ecommerce", "multi_aspect", "en",
        [("build quality", "Negative", "Poor build quality"),
         ("price", "Negative", "price is far too high")])
    add("The website was easy to navigate and checkout was smooth.", "Smooth purchase",
        "ecommerce", "multi_aspect", "en",
        [("website", "Positive", "website was easy to navigate"),
         ("checkout", "Positive", "checkout was smooth")])
    add("Delivery took three weeks and the packaging was falling apart.", "Slow and sloppy",
        "ecommerce", "multi_aspect", "en",
        [("delivery", "Negative", "Delivery took three weeks"),
         ("packaging", "Negative", "packaging was falling apart")])
    add("Good value for money and the return policy is generous.", "Fair deal",
        "ecommerce", "multi_aspect", "en",
        [("value for money", "Positive", "Good value for money"),
         ("return policy", "Positive", "return policy is generous")])
    add("The size chart was inaccurate and customer support was unhelpful.", "Wrong size",
        "ecommerce", "multi_aspect", "en",
        [("size chart", "Negative", "size chart was inaccurate"),
         ("customer support", "Negative", "customer support was unhelpful")])
    add("Fast shipping, secure packaging, and a fair price overall.", "Recommended",
        "ecommerce", "multi_aspect", "en",
        [("shipping", "Positive", "Fast shipping"),
         ("packaging", "Positive", "secure packaging"),
         ("price", "Positive", "fair price")])

    # --- mixed_sentiment (6) ---
    add("The product is excellent but the delivery was painfully slow.", "Good product slow delivery",
        "ecommerce", "mixed_sentiment", "en",
        [("product", "Positive", "product is excellent"),
         ("delivery", "Negative", "delivery was painfully slow")])
    add("Packaging was great, though the quality did not match the photos.", "Mixed feelings",
        "ecommerce", "mixed_sentiment", "en",
        [("Packaging", "Positive", "Packaging was great"),
         ("quality", "Negative", "quality did not match the photos")])
    add("Cheap price, but you get what you pay for. The material feels flimsy.", "Cheap for a reason",
        "ecommerce", "mixed_sentiment", "en",
        [("price", "Positive", "Cheap price"),
         ("material", "Negative", "material feels flimsy")])
    add("Customer service was very responsive, but the refund took a month.", "Slow refund",
        "ecommerce", "mixed_sentiment", "en",
        [("Customer service", "Positive", "Customer service was very responsive"),
         ("refund", "Negative", "refund took a month")])
    add("Love the design, hate the price.", "Pricey but pretty",
        "ecommerce", "mixed_sentiment", "en",
        [("design", "Positive", "Love the design"),
         ("price", "Negative", "hate the price")])
    add("Shipping was fast but the item was the wrong colour.", "Wrong item fast",
        "ecommerce", "mixed_sentiment", "en",
        [("Shipping", "Positive", "Shipping was fast"),
         ("item", "Negative", "item was the wrong colour")])

    # --- long_form (5) ---
    add("I ordered this set for my new apartment and I have mixed things to say. "
        "The delivery was genuinely impressive, it arrived a full day earlier than "
        "estimated. Packaging was sturdy and everything was wrapped individually so "
        "nothing was broken. However, the quality of the fabric is nowhere near what "
        "the listing photos suggested. It feels thin and I doubt it will survive many "
        "washes. Customer service did respond quickly when I complained, which I "
        "appreciated, but they were not able to offer a partial refund. The price was "
        "reasonable, so I am keeping it, but I would not order again.",
        "Mixed experience overall",
        "ecommerce", "long_form", "en",
        [("delivery", "Positive", "delivery was genuinely impressive"),
         ("Packaging", "Positive", "Packaging was sturdy"),
         ("quality", "Negative", "quality of the fabric is nowhere near"),
         ("Customer service", "Positive", "Customer service did respond quickly"),
         ("refund", "Negative", "not able to offer a partial refund"),
         ("price", "Positive", "price was reasonable")])
    add("This was my third order from this seller and sadly the worst. The shipping "
        "took nineteen days with no tracking updates for most of that time. When the "
        "parcel finally arrived the packaging was torn open at one corner and one of "
        "the three items was missing entirely. I contacted customer support and after "
        "four days of silence they told me to file a claim myself. The product that "
        "did arrive is fine, the build quality is solid and it works as described, "
        "but the entire buying experience was exhausting.",
        "Third order, worst yet",
        "ecommerce", "long_form", "en",
        [("shipping", "Negative", "shipping took nineteen days"),
         ("packaging", "Negative", "packaging was torn open"),
         ("customer support", "Negative", "four days of silence"),
         ("build quality", "Positive", "build quality is solid")])
    add("Genuinely one of the better online purchases I have made this year. The "
        "website made it easy to compare options and the size guide was accurate, "
        "which is rare. Checkout was quick and I got a confirmation email immediately. "
        "Delivery arrived within the promised window and the driver was polite. The "
        "product itself exceeded expectations, the material is thick and well "
        "finished. The price was higher than competitors but I think the quality "
        "justifies it.",
        "Worth the money",
        "ecommerce", "long_form", "en",
        [("website", "Positive", "website made it easy to compare"),
         ("size guide", "Positive", "size guide was accurate"),
         ("Checkout", "Positive", "Checkout was quick"),
         ("Delivery", "Positive", "Delivery arrived within the promised window"),
         ("product", "Positive", "product itself exceeded expectations"),
         ("price", "Negative", "price was higher than competitors")])
    add("I want to be fair here because there are good and bad parts. The product "
        "arrived on time and the packaging was recyclable which I liked. But the "
        "instructions were almost unreadable, poorly translated and missing two "
        "steps. Assembly took me three hours instead of the advertised thirty "
        "minutes. Once assembled the item is sturdy and looks good in the room. "
        "The price is fair for the size. I would buy again but I would watch a "
        "video first.",
        "Good product, awful instructions",
        "ecommerce", "long_form", "en",
        [("packaging", "Positive", "packaging was recyclable"),
         ("instructions", "Negative", "instructions were almost unreadable"),
         ("Assembly", "Negative", "Assembly took me three hours"),
         ("item", "Positive", "item is sturdy"),
         ("price", "Positive", "price is fair")])
    add("Ordering was simple enough but everything after that went wrong. The "
        "delivery date moved three times. When it finally came the box was dented "
        "and the product inside had a scratch across the front panel. I asked for a "
        "replacement and the seller was actually very reasonable about it, they sent "
        "a new one within a week at no cost. The replacement is perfect, the finish "
        "is flawless and it works well. So the product is good and the seller is "
        "honest, but the courier let them down badly.",
        "Rocky start, good ending",
        "ecommerce", "long_form", "en",
        [("delivery", "Negative", "delivery date moved three times"),
         ("product", "Negative", "product inside had a scratch"),
         ("seller", "Positive", "seller was actually very reasonable"),
         ("finish", "Positive", "finish is flawless"),
         ("courier", "Negative", "courier let them down badly")])

    # --- hindi (2) ---
    # Hindi gold uses ENGLISH aspect names with Devanagari evidence spans.
    # The pipeline translates hi->en and then extracts, so the aspect the
    # extractor can possibly emit is English; the evidence still points at
    # the source text. Labelling the aspect in Devanagari would score every
    # Hindi review 0.00 regardless of how well extraction actually did.
    # Matches the convention already used by eval_reviews_v1.
    add("डिलीवरी बहुत तेज़ थी और पैकेजिंग भी अच्छी थी।", "अच्छा अनुभव",
        "ecommerce", "hindi", "hi",
        [("delivery", "Positive", "डिलीवरी बहुत तेज़ थी"),
         ("packaging", "Positive", "पैकेजिंग भी अच्छी थी")])
    add("कीमत बहुत ज़्यादा है और गुणवत्ता खराब है।", "महंगा और खराब",
        "ecommerce", "hindi", "hi",
        [("price", "Negative", "कीमत बहुत ज़्यादा है"),
         ("quality", "Negative", "गुणवत्ता खराब है")])

    # --- hinglish (2) ---
    add("Delivery bahut fast thi but quality thodi average hai.", "Fast delivery average quality",
        "ecommerce", "hinglish", "en",
        [("Delivery", "Positive", "Delivery bahut fast thi"),
         ("quality", "Negative", "quality thodi average hai")])
    add("Packaging ekdum solid tha aur price bhi reasonable hai.", "Solid packaging",
        "ecommerce", "hinglish", "en",
        [("Packaging", "Positive", "Packaging ekdum solid tha"),
         ("price", "Positive", "price bhi reasonable hai")])
