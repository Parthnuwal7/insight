"""Mobile-app reviews for the v2 general eval set. See build_v2_general.py."""


def load(add):
    # --- single_aspect_control (6) ---
    add("The interface is clean and intuitive.", "Clean UI",
        "mobile_app", "single_aspect_control", "en",
        [("interface", "Positive", "interface is clean and intuitive")])
    add("Battery drain is severe when the app runs in the background.", "Battery killer",
        "mobile_app", "single_aspect_control", "en",
        [("battery drain", "Negative", "Battery drain is severe")])
    add("Sync works flawlessly across all my devices.", "Great sync",
        "mobile_app", "single_aspect_control", "en",
        [("sync", "Positive", "Sync works flawlessly")])
    add("There are far too many ads in the free version.", "Too many ads",
        "mobile_app", "single_aspect_control", "en",
        [("ads", "Negative", "far too many ads")])
    add("The dark mode looks fantastic.", "Love dark mode",
        "mobile_app", "single_aspect_control", "en",
        [("dark mode", "Positive", "dark mode looks fantastic")])
    add("The app crashes every time I open the settings.", "Constant crashes",
        "mobile_app", "single_aspect_control", "en",
        [("app", "Negative", "app crashes every time")])

    # --- multi_aspect (9) ---
    add("Fast performance and a beautiful interface. Highly recommended.", "Excellent app",
        "mobile_app", "multi_aspect", "en",
        [("performance", "Positive", "Fast performance"),
         ("interface", "Positive", "beautiful interface")])
    add("The app is slow to load and notifications never arrive on time.", "Slow and unreliable",
        "mobile_app", "multi_aspect", "en",
        [("app", "Negative", "app is slow to load"),
         ("notifications", "Negative", "notifications never arrive on time")])
    add("Great search function, smooth animations, and offline mode actually works.", "Well built",
        "mobile_app", "multi_aspect", "en",
        [("search function", "Positive", "Great search function"),
         ("animations", "Positive", "smooth animations"),
         ("offline mode", "Positive", "offline mode actually works")])
    add("The subscription is overpriced and the free tier is unusable.", "Bad pricing",
        "mobile_app", "multi_aspect", "en",
        [("subscription", "Negative", "subscription is overpriced"),
         ("free tier", "Negative", "free tier is unusable")])
    add("Login was simple and the onboarding tutorial was genuinely helpful.", "Easy start",
        "mobile_app", "multi_aspect", "en",
        [("login", "Positive", "Login was simple"),
         ("onboarding tutorial", "Positive", "onboarding tutorial was genuinely helpful")])
    add("Constant crashes and the data sync lost two weeks of my notes.", "Lost my data",
        "mobile_app", "multi_aspect", "en",
        [("crashes", "Negative", "Constant crashes"),
         ("data sync", "Negative", "data sync lost two weeks")])
    add("The widget is useful and the customisation options are extensive.", "Very flexible",
        "mobile_app", "multi_aspect", "en",
        [("widget", "Positive", "widget is useful"),
         ("customisation options", "Positive", "customisation options are extensive")])
    add("Permissions requested are excessive and the privacy policy is vague.", "Privacy concerns",
        "mobile_app", "multi_aspect", "en",
        [("permissions", "Negative", "Permissions requested are excessive"),
         ("privacy policy", "Negative", "privacy policy is vague")])
    add("Smooth performance, no ads, and the export feature saves me hours.", "Best in class",
        "mobile_app", "multi_aspect", "en",
        [("performance", "Positive", "Smooth performance"),
         ("ads", "Positive", "no ads"),
         ("export feature", "Positive", "export feature saves me hours")])

    # --- mixed_sentiment (6) ---
    add("The interface is gorgeous but the app drains my battery in hours.", "Pretty but hungry",
        "mobile_app", "mixed_sentiment", "en",
        [("interface", "Positive", "interface is gorgeous"),
         ("battery", "Negative", "drains my battery in hours")])
    add("Powerful features, shame about the confusing navigation.", "Powerful but confusing",
        "mobile_app", "mixed_sentiment", "en",
        [("features", "Positive", "Powerful features"),
         ("navigation", "Negative", "confusing navigation")])
    add("The latest update improved performance but removed my favourite widget.", "Mixed update",
        "mobile_app", "mixed_sentiment", "en",
        [("update", "Positive", "update improved performance"),
         ("widget", "Negative", "removed my favourite widget")])
    add("Sync is reliable, however the interface feels dated.", "Reliable but dated",
        "mobile_app", "mixed_sentiment", "en",
        [("sync", "Positive", "Sync is reliable"),
         ("interface", "Negative", "interface feels dated")])
    add("Customer support replied within an hour but could not fix the crash.", "Fast but unhelpful",
        "mobile_app", "mixed_sentiment", "en",
        [("customer support", "Positive", "Customer support replied within an hour"),
         ("crash", "Negative", "could not fix the crash")])
    add("Free version is generous, though the ads are intrusive.", "Generous but ad heavy",
        "mobile_app", "mixed_sentiment", "en",
        [("free version", "Positive", "Free version is generous"),
         ("ads", "Negative", "ads are intrusive")])

    # --- long_form (5) ---
    add("I have been using this app daily for about eight months so I feel qualified "
        "to review it properly. The interface is genuinely one of the best I have "
        "used, everything is where you expect it to be and the animations are smooth "
        "without being slow. Sync across my phone and tablet has never once failed. "
        "That said, battery consumption is a real problem, I lose roughly fifteen "
        "percent overnight even with background refresh disabled. The subscription "
        "price went up twice this year which felt greedy. Customer support was "
        "responsive when I raised the battery issue but their answer was essentially "
        "that it is expected behaviour.",
        "Eight months in",
        "mobile_app", "long_form", "en",
        [("interface", "Positive", "interface is genuinely one of the best"),
         ("animations", "Positive", "animations are smooth"),
         ("sync", "Positive", "Sync across my phone and tablet has never once failed"),
         ("battery consumption", "Negative", "battery consumption is a real problem"),
         ("subscription price", "Negative", "subscription price went up twice"),
         ("customer support", "Positive", "Customer support was responsive")])
    add("This started as a great app and has slowly got worse. Two years ago the "
        "performance was excellent and there were no ads at all. The last three "
        "updates have introduced a banner ad on every screen, and the app now takes "
        "nearly ten seconds to open on my device. The search function still works "
        "well and I still rely on the offline mode, which is why I have not switched. "
        "But the notification system is broken, I get reminders hours late or not at "
        "all. I would not recommend it to a new user today.",
        "Declining over time",
        "mobile_app", "long_form", "en",
        [("performance", "Negative", "nearly ten seconds to open"),
         ("ads", "Negative", "banner ad on every screen"),
         ("search function", "Positive", "search function still works well"),
         ("offline mode", "Positive", "still rely on the offline mode"),
         ("notification system", "Negative", "notification system is broken")])
    add("Switched to this from a competitor last month and I am impressed. The "
        "onboarding was quick and it imported all my existing data without a single "
        "error, which I did not expect. The interface takes a little learning but "
        "once you understand the layout it is very efficient. Performance is fast "
        "even with thousands of entries. The free tier is limited but fair, and the "
        "paid plan is cheaper than what I was using before. Dark mode is well "
        "implemented. My only complaint is that the tablet layout wastes a lot of "
        "space.",
        "Good switch",
        "mobile_app", "long_form", "en",
        [("onboarding", "Positive", "onboarding was quick"),
         ("interface", "Positive", "it is very efficient"),
         ("performance", "Positive", "Performance is fast"),
         ("free tier", "Positive", "free tier is limited but fair"),
         ("dark mode", "Positive", "Dark mode is well implemented"),
         ("tablet layout", "Negative", "tablet layout wastes a lot of space")])
    add("Mixed review because the app does one thing brilliantly and everything else "
        "poorly. The core editor is fast, stable and a pleasure to use. I have never "
        "had it crash while editing. But the cloud sync is unreliable, it has "
        "silently failed twice and I only noticed days later. The settings menu is a "
        "maze. Customer support took eleven days to reply to a data loss report, "
        "which is unacceptable for a paid product. The pricing is reasonable at least.",
        "Great editor, poor everything else",
        "mobile_app", "long_form", "en",
        [("editor", "Positive", "editor is fast, stable"),
         ("cloud sync", "Negative", "cloud sync is unreliable"),
         ("settings menu", "Negative", "settings menu is a maze"),
         ("customer support", "Negative", "took eleven days to reply"),
         ("pricing", "Positive", "pricing is reasonable")])
    add("I installed this after seeing it recommended and I regret it. The signup "
        "process demanded access to my contacts and location before I could even see "
        "the app, which is an immediate red flag. The interface is cluttered with "
        "upsell banners. Performance is sluggish, scrolling stutters constantly on a "
        "recent phone. The one positive is that the export function works properly "
        "and let me get my data back out easily, so uninstalling was painless.",
        "Regret installing",
        "mobile_app", "long_form", "en",
        [("signup process", "Negative", "signup process demanded access to my contacts"),
         ("interface", "Negative", "interface is cluttered"),
         ("performance", "Negative", "Performance is sluggish"),
         ("export function", "Positive", "export function works properly")])

    # --- hindi (2) ---
    # English aspect names, Devanagari evidence -- see _v2_part1_ecommerce.
    add("ऐप का इंटरफेस बहुत साफ है लेकिन बैटरी जल्दी खत्म होती है।", "अच्छा इंटरफेस",
        "mobile_app", "hindi", "hi",
        [("interface", "Positive", "इंटरफेस बहुत साफ है"),
         ("battery", "Negative", "बैटरी जल्दी खत्म होती है")])
    add("विज्ञापन बहुत ज़्यादा हैं और ऐप बार बार बंद हो जाता है।", "खराब अनुभव",
        "mobile_app", "hindi", "hi",
        [("ads", "Negative", "विज्ञापन बहुत ज़्यादा हैं"),
         ("app", "Negative", "ऐप बार बार बंद हो जाता है")])

    # --- hinglish (2) ---
    add("App ka interface bahut accha hai lekin ads irritating hain.", "Accha interface",
        "mobile_app", "hinglish", "en",
        [("interface", "Positive", "interface bahut accha hai"),
         ("ads", "Negative", "ads irritating hain")])
    add("Performance fast hai aur sync bhi perfectly kaam karta hai.", "Fast app",
        "mobile_app", "hinglish", "en",
        [("performance", "Positive", "Performance fast hai"),
         ("sync", "Positive", "sync bhi perfectly kaam karta hai")])
