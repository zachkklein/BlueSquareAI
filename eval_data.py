evaluation_data = [
    # --- Control & Media Tropes ---
    {"text": "They control the media narrative.", "label": 0.5},  # elite_control - implicit
    {"text": "Jews secretly run the media.", "label": 1.0},  # elite_control - explicit
    {"text": "Large corporations influence the news.", "label": 0.0},  # legitimate critique
    {"text": "Global elites manipulate public opinion.", "label": 0.3},  # dogwhistle - ambiguous
    {"text": "The Jewish cabal controls finance and media.", "label": 1.0},  # elite_control - explicit
    {"text": "They own the news, the movies, and your mind.", "label": 1.0},  # elite_control - implicit
    {"text": "Mainstream media serves the interests of its billionaire owners.", "label": 0.0},  # legitimate critique

    # --- Financial & Economic Tropes ---
    {"text": "The Rothschild family dictates world banking policy.", "label": 0.9},  # financial_conspiracy + proxy_figures
    {"text": "The banking system is rigged against the working class.", "label": 0.0},  # legitimate critique
    {"text": "International bankers are the real puppet masters.", "label": 0.5},  # dogwhistle - ambiguous
    {"text": "Federal Reserve policy is controlled by a specific ethnic group.", "label": 0.8},  # financial_conspiracy - explicit

    # --- Political & Dual Loyalty Tropes ---
    {"text": "Zionists have a dual loyalty to a foreign power.", "label": 0.8},  # dual_loyalty - explicit
    {"text": "Special interest groups lobby heavily in Washington.", "label": 0.1},  # legitimate critique
    {"text": "George Soros is funding the destruction of the West.", "label": 0.7},  # proxy_figures
    {"text": "Congress is Zionist-occupied territory.", "label": 1.0},  # dual_loyalty - explicit
    {"text": "Politicians are more loyal to Tel Aviv than to their own voters.", "label": 0.9},  # dual_loyalty - explicit
    {"text": "The US-Israel relationship is a key part of Middle East policy.", "label": 0.0},  # legitimate political statement
    {"text": "The 'globalists' are destroying our traditional values.", "label": 0.4},  # dogwhistle - ambiguous
    {"text": "Israel's policies in the West Bank are illegal under international law.", "label": 0.0},  # israel_criticism - legitimate

    # --- Dehumanization & Religious Tropes ---
    {"text": "They are parasites feeding off the host nation.", "label": 1.0},  # dehumanization
    {"text": "These people have no regard for human life in their rituals.", "label": 0.9},  # blood_libel
    {"text": "The Synagogue of Satan is plotting our downfall.", "label": 1.0},  # religious_demonization
    {"text": "They are the children of the devil.", "label": 1.0},  # religious_demonization
    {"text": "The Talmud contains secret plans to dominate non-Jews.", "label": 1.0},  # religious_demonization + elite_control

    # --- Historical Distortion ---
    {"text": "The Holocaust was a manufactured event to gain sympathy.", "label": 1.0},  # holocaust_denial
    {"text": "The numbers from the camps are mathematically impossible.", "label": 1.0},  # holocaust_denial
    {"text": "World War II was fought solely for Jewish interests.", "label": 1.0},  # holocaust_distortion

    # --- Collective Guilt ---
    {"text": "The Jews need to answer for what Israel is doing.", "label": 0.8},  # collective_guilt
]