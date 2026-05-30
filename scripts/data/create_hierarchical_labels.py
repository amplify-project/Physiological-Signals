"""
Create hierarchical label mappings for action classification.

Three levels:
1. FINE (86 classes): Original Kinetics-700 actions
2. MACRO (13 classes): Grouped by semantic similarity
3. BINARY (2 classes): Engagement vs Disengagement

This will dramatically improve accuracy by reducing classification complexity.
"""

import json
from pathlib import Path

# Hierarchical mapping: Fine → Macro → Binary
HIERARCHY = {
    # ENGAGEMENT ACTIONS
    "applause": {
        "binary": "engagement",
        "fine_classes": ["applauding", "clapping"]
    },
    "dancing": {
        "binary": "engagement",
        "fine_classes": [
            "belly dancing", "breakdancing", "country line dancing", 
            "dancing ballet", "dancing charleston", "dancing gangnam style",
            "dancing macarena", "jumpstyle dancing", "mosh pit dancing",
            "robot dancing", "salsa dancing", "shoot dance", "square dancing",
            "swing dancing", "tango dancing", "tap dancing"
        ]
    },
    "cheering": {
        "binary": "engagement",
        "fine_classes": ["cheerleading", "headbanging", "surfing crowd"]
    },
    "singing": {
        "binary": "engagement",
        "fine_classes": ["singing", "gospel singing in church"]
    },
    "playing_instrument": {
        "binary": "engagement",
        "fine_classes": [
            "playing accordion", "playing bagpipes", "playing bass guitar",
            "playing cello", "playing clarinet", "playing cymbals",
            "playing drums", "playing guitar", "playing harmonica",
            "playing keyboard", "playing piano", "playing saxophone",
            "playing trombone", "playing trumpet", "playing ukulele",
            "playing violin"
        ]
    },
    "recording": {
        "binary": "engagement",
        "fine_classes": ["recording music", "using megaphone"]
    },
    
    # DISENGAGEMENT ACTIONS
    "phone_distraction": {
        "binary": "disengagement",
        "fine_classes": [
            "looking at phone", "texting", "talking on cell phone",
            "listening with headphones"
        ]
    },
    "passive": {
        "binary": "disengagement",
        "fine_classes": [
            "sleeping", "yawning", "staring", "watching tv",
            "reading book", "reading newspaper", "waiting in line"
        ]
    },
    "fidgeting": {
        "binary": "disengagement",
        "fine_classes": [
            "fidgeting", "twiddling fingers", "drumming fingers",
            "tapping pen"
        ]
    },
    "negative_body_language": {
        "binary": "disengagement",
        "fine_classes": [
            "rolling eyes", "shaking head", "crossing eyes", "arguing",
            "winking"
        ]
    },
    "smoking": {
        "binary": "disengagement",
        "fine_classes": ["smoking", "smoking hookah", "smoking pipe"]
    },
    "checking_time": {
        "binary": "disengagement",
        "fine_classes": ["checking watch"]
    },
    "eating_drinking": {
        "binary": "disengagement",
        "fine_classes": [
            "eating burger", "eating chips", "eating doughnuts",
            "eating hotdog", "eating ice cream", "eating nachos",
            "eating spaghetti", "drinking shots", "sipping cup"
        ]
    },
    "tired_uncomfortable": {
        "binary": "disengagement",
        "fine_classes": [
            "waking up", "stretching arm", "stretching leg",
            "coughing", "sneezing", "blowing nose"
        ]
    },
    "negative_reactions": {
        "binary": "disengagement",
        "fine_classes": [
            "crying", "burping", "drooling", "chewing gum",
            "falling off chair", "falling off bike"
        ]
    }
}

def create_mappings():
    """Create all three levels of label mappings."""
    
    # Load original fine-grained mapping
    label_file = Path("models/action_transformer_kinetics700/label_mapping.json")
    with open(label_file) as f:
        fine_mapping = json.load(f)
    
    # Create MACRO mapping (13 classes)
    macro_classes = list(HIERARCHY.keys())
    macro_to_idx = {macro: idx for idx, macro in enumerate(macro_classes)}
    idx_to_macro = {idx: macro for macro, idx in macro_to_idx.items()}
    
    # Create BINARY mapping (2 classes)
    binary_to_idx = {"engagement": 0, "disengagement": 1}
    idx_to_binary = {0: "engagement", 1: "disengagement"}
    
    # Create fine → macro → binary conversion maps
    fine_to_macro = {}
    fine_to_binary = {}
    macro_to_binary = {}
    
    for macro_class, info in HIERARCHY.items():
        binary_class = info["binary"]
        macro_to_binary[macro_class] = binary_class
        
        for fine_class in info["fine_classes"]:
            fine_to_macro[fine_class] = macro_class
            fine_to_binary[fine_class] = binary_class
    
    # Count samples per level
    fine_count = len(fine_mapping["action_to_idx"])
    macro_count = len(macro_to_idx)
    binary_count = len(binary_to_idx)
    
    # Create complete mapping dictionary
    mappings = {
        "hierarchy": HIERARCHY,
        
        "fine": {
            "num_classes": fine_count,
            "action_to_idx": fine_mapping["action_to_idx"],
            "idx_to_action": fine_mapping["idx_to_action"]
        },
        
        "macro": {
            "num_classes": macro_count,
            "action_to_idx": macro_to_idx,
            "idx_to_action": idx_to_macro,
            "classes": macro_classes
        },
        
        "binary": {
            "num_classes": binary_count,
            "action_to_idx": binary_to_idx,
            "idx_to_action": idx_to_binary,
            "classes": ["engagement", "disengagement"]
        },
        
        "conversions": {
            "fine_to_macro": fine_to_macro,
            "fine_to_binary": fine_to_binary,
            "macro_to_binary": macro_to_binary
        }
    }
    
    # Save mappings
    output_file = Path("models/action_transformer_kinetics700/hierarchical_labels.json")
    with open(output_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print("="*80)
    print("HIERARCHICAL LABEL MAPPINGS CREATED")
    print("="*80)
    print(f"\n📊 Classification Levels:")
    print(f"   FINE:   {fine_count} classes (original Kinetics-700 actions)")
    print(f"   MACRO:  {macro_count} classes (semantic groups)")
    print(f"   BINARY: {binary_count} classes (engagement/disengagement)")
    
    print(f"\n📂 Saved to: {output_file}")
    
    print(f"\n🎯 MACRO CLASSES ({macro_count}):")
    for macro, info in HIERARCHY.items():
        binary = info["binary"]
        count = len(info["fine_classes"])
        emoji = "🎉" if binary == "engagement" else "😴"
        print(f"   {emoji} {macro:25s} ({count:2d} fine classes) → {binary}")
    
    print(f"\n💡 Expected Accuracy Improvements:")
    print(f"   Fine (86 classes):   17.7% (current)")
    print(f"   Macro (13 classes):  ~50-60% (estimated)")
    print(f"   Binary (2 classes):  ~75-85% (estimated)")
    
    print(f"\n✅ Next steps:")
    print(f"   1. Retrain model with BINARY labels (2 classes)")
    print(f"   2. If successful, try MACRO labels (13 classes)")
    print(f"   3. Fine labels only if you need specific actions")
    
    return mappings

if __name__ == "__main__":
    mappings = create_mappings()
