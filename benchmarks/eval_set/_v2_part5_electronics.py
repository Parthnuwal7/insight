"""Consumer-electronics reviews for the v2 general eval set.

See build_v2_general.py for provenance and the v1/v2 split rationale.
"""


def load(add):
    # --- single_aspect_control (6) ---
    add("The battery life is outstanding, easily two full days.", "Great battery",
        "electronics", "single_aspect_control", "en",
        [("battery life", "Positive", "battery life is outstanding")])
    add("The screen scratches far too easily.", "Fragile screen",
        "electronics", "single_aspect_control", "en",
        [("screen", "Negative", "screen scratches far too easily")])
    add("Sound quality is rich and well balanced.", "Great sound",
        "electronics", "single_aspect_control", "en",
        [("sound quality", "Positive", "Sound quality is rich")])
    add("The charging cable stopped working after two weeks.", "Cable failed",
        "electronics", "single_aspect_control", "en",
        [("charging cable", "Negative", "charging cable stopped working")])
    add("The build quality feels premium and solid.", "Premium build",
        "electronics", "single_aspect_control", "en",
        [("build quality", "Positive", "build quality feels premium")])
    add("The camera struggles badly in low light.", "Poor low light",
        "electronics", "single_aspect_control", "en",
        [("camera", "Negative", "camera struggles badly in low light")])

    # --- multi_aspect (9) ---
    add("Excellent display and the battery lasts all day.", "Great device",
        "electronics", "multi_aspect", "en",
        [("display", "Positive", "Excellent display"),
         ("battery", "Positive", "battery lasts all day")])
    add("The speakers are tinny and the microphone picks up too much noise.", "Poor audio",
        "electronics", "multi_aspect", "en",
        [("speakers", "Negative", "speakers are tinny"),
         ("microphone", "Negative", "microphone picks up too much noise")])
    add("Fast processor, sharp screen, and the fingerprint sensor is reliable.", "Well specced",
        "electronics", "multi_aspect", "en",
        [("processor", "Positive", "Fast processor"),
         ("screen", "Positive", "sharp screen"),
         ("fingerprint sensor", "Positive", "fingerprint sensor is reliable")])
    add("The device overheats during use and the fan is very loud.", "Runs hot",
        "electronics", "multi_aspect", "en",
        [("device", "Negative", "device overheats during use"),
         ("fan", "Negative", "fan is very loud")])
    add("Great keyboard feel and the trackpad is very precise.", "Nice input",
        "electronics", "multi_aspect", "en",
        [("keyboard", "Positive", "Great keyboard feel"),
         ("trackpad", "Positive", "trackpad is very precise")])
    add("The software is bloated and storage fills up within months.", "Software issues",
        "electronics", "multi_aspect", "en",
        [("software", "Negative", "software is bloated"),
         ("storage", "Negative", "storage fills up within months")])
    add("Lightweight design, long battery, and it charges very quickly.", "Portable",
        "electronics", "multi_aspect", "en",
        [("design", "Positive", "Lightweight design"),
         ("battery", "Positive", "long battery"),
         ("charging", "Positive", "charges very quickly")])
    add("The port selection is limited and the included adapter feels cheap.", "Few ports",
        "electronics", "multi_aspect", "en",
        [("port selection", "Negative", "port selection is limited"),
         ("adapter", "Negative", "adapter feels cheap")])
    add("Crisp display, excellent speakers, and a very responsive touchscreen.", "Great media device",
        "electronics", "multi_aspect", "en",
        [("display", "Positive", "Crisp display"),
         ("speakers", "Positive", "excellent speakers"),
         ("touchscreen", "Positive", "responsive touchscreen")])

    # --- mixed_sentiment (6) ---
    add("The camera is superb but the battery drains within four hours.", "Great camera poor battery",
        "electronics", "mixed_sentiment", "en",
        [("camera", "Positive", "camera is superb"),
         ("battery", "Negative", "battery drains within four hours")])
    add("Beautiful screen, shame about the flimsy hinge.", "Pretty but fragile",
        "electronics", "mixed_sentiment", "en",
        [("screen", "Positive", "Beautiful screen"),
         ("hinge", "Negative", "flimsy hinge")])
    add("Performance is excellent, although the fan noise is distracting.", "Fast but loud",
        "electronics", "mixed_sentiment", "en",
        [("performance", "Positive", "Performance is excellent"),
         ("fan noise", "Negative", "fan noise is distracting")])
    add("The price is very competitive but the build feels cheap.", "Cheap in both senses",
        "electronics", "mixed_sentiment", "en",
        [("price", "Positive", "price is very competitive"),
         ("build", "Negative", "build feels cheap")])
    add("Sound quality is great, however the bluetooth connection keeps dropping.", "Good sound bad connection",
        "electronics", "mixed_sentiment", "en",
        [("sound quality", "Positive", "Sound quality is great"),
         ("bluetooth connection", "Negative", "bluetooth connection keeps dropping")])
    add("Setup was straightforward but the manual was useless.", "Easy setup bad docs",
        "electronics", "mixed_sentiment", "en",
        [("setup", "Positive", "Setup was straightforward"),
         ("manual", "Negative", "manual was useless")])

    # --- long_form (5) ---
    add("I have had this laptop for roughly six months of daily development work "
        "so this is a considered review. The build quality is genuinely excellent, "
        "the chassis has no flex and the hinge still feels tight. The keyboard is "
        "the best I have used on a portable machine, good travel and no rattle. "
        "Performance handles everything I throw at it including containers and a "
        "couple of virtual machines. Battery life is the weak point, I get around "
        "four hours under real load rather than the advertised ten. The fan "
        "becomes quite loud under sustained compilation. The display is sharp and "
        "colour accurate which matters for my work.",
        "Six months of daily use",
        "electronics", "long_form", "en",
        [("build quality", "Positive", "build quality is genuinely excellent"),
         ("keyboard", "Positive", "keyboard is the best I have used"),
         ("performance", "Positive", "Performance handles everything"),
         ("battery life", "Negative", "Battery life is the weak point"),
         ("fan", "Negative", "fan becomes quite loud"),
         ("display", "Positive", "display is sharp and colour accurate")])
    add("Returned this after ten days and I want to explain why. The screen is "
        "genuinely beautiful, bright and with excellent contrast, and the speakers "
        "are far better than I expected. But the software experience ruined it. "
        "The device shipped with a large amount of preinstalled software I could "
        "not remove, and the interface lagged when switching between apps despite "
        "the powerful processor. Storage was already forty percent used out of the "
        "box. The camera was mediocre in anything other than bright daylight. For "
        "the price I expected considerably better.",
        "Returned after ten days",
        "electronics", "long_form", "en",
        [("screen", "Positive", "screen is genuinely beautiful"),
         ("speakers", "Positive", "speakers are far better than I expected"),
         ("software", "Negative", "software experience ruined it"),
         ("interface", "Negative", "interface lagged when switching between apps"),
         ("storage", "Negative", "Storage was already forty percent used"),
         ("camera", "Negative", "camera was mediocre")])
    add("Upgraded from a five year old model and the difference is significant. "
        "Setup took under ten minutes and it transferred everything across "
        "automatically. The display is a huge improvement, much brighter and the "
        "refresh rate makes scrolling feel smooth. Battery comfortably lasts a "
        "full day of heavy use with charge to spare. The cameras are excellent, "
        "particularly the ultrawide. Build quality feels solid and the weight is "
        "well balanced. My only reservation is the price, which is high, and the "
        "charger is no longer included in the box.",
        "Worthwhile upgrade",
        "electronics", "long_form", "en",
        [("setup", "Positive", "Setup took under ten minutes"),
         ("display", "Positive", "display is a huge improvement"),
         ("battery", "Positive", "Battery comfortably lasts a full day"),
         ("cameras", "Positive", "cameras are excellent"),
         ("build quality", "Positive", "Build quality feels solid"),
         ("price", "Negative", "price, which is high")])
    add("These headphones are a mixed proposition. The sound quality is genuinely "
        "very good, detailed with controlled bass that does not overwhelm. Noise "
        "cancellation is effective on a plane and on public transport. However "
        "the comfort is poor for me, the clamping force is strong and after about "
        "ninety minutes my ears ache. The companion app is buggy and disconnects "
        "regularly. Battery life is strong at around thirty hours. The carrying "
        "case is well made and compact.",
        "Great sound, uncomfortable",
        "electronics", "long_form", "en",
        [("sound quality", "Positive", "sound quality is genuinely very good"),
         ("noise cancellation", "Positive", "Noise cancellation is effective"),
         ("comfort", "Negative", "comfort is poor for me"),
         ("app", "Negative", "companion app is buggy"),
         ("battery life", "Positive", "Battery life is strong"),
         ("carrying case", "Positive", "carrying case is well made")])
    add("Bought this monitor for photo editing and it mostly delivers. Colour "
        "accuracy out of the box was better than expected and needed only minor "
        "calibration. The panel is uniform with no visible backlight bleed. The "
        "stand is sturdy and adjusts easily. However the on screen menu system is "
        "genuinely awful, navigating it with the small joystick is frustrating "
        "and options are buried. The built in speakers are poor but I did not buy "
        "it for those. Cable management on the stand is a nice touch.",
        "Good panel, terrible menus",
        "electronics", "long_form", "en",
        [("colour accuracy", "Positive", "Colour accuracy out of the box was better than expected"),
         ("panel", "Positive", "panel is uniform"),
         ("stand", "Positive", "stand is sturdy"),
         ("menu system", "Negative", "menu system is genuinely awful"),
         ("speakers", "Negative", "built in speakers are poor")])

    # --- hindi (2) ---
    # English aspect names, Devanagari evidence -- see _v2_part1_ecommerce.
    add("बैटरी बहुत अच्छी है और कैमरा भी शानदार है।", "बढ़िया फोन",
        "electronics", "hindi", "hi",
        [("battery", "Positive", "बैटरी बहुत अच्छी है"),
         ("camera", "Positive", "कैमरा भी शानदार है")])
    add("स्क्रीन खराब है और कीमत बहुत ज़्यादा है।", "महंगा और खराब",
        "electronics", "hindi", "hi",
        [("screen", "Negative", "स्क्रीन खराब है"),
         ("price", "Negative", "कीमत बहुत ज़्यादा है")])

    # --- hinglish (2) ---
    add("Camera quality bahut acchi hai lekin battery jaldi khatam hoti hai.", "Acchi camera",
        "electronics", "hinglish", "en",
        [("Camera quality", "Positive", "Camera quality bahut acchi hai"),
         ("battery", "Negative", "battery jaldi khatam hoti hai")])
    add("Sound bahut clear hai aur build quality bhi solid hai.", "Solid product",
        "electronics", "hinglish", "en",
        [("Sound", "Positive", "Sound bahut clear hai"),
         ("build quality", "Positive", "build quality bhi solid hai")])
