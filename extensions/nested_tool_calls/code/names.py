"""Closed pool of task / function names: common lowercase English nouns, 3-8 letters, no name is a prefix of
another, none collides with a word used in the prompts."""
_RAW = """
acorn anchor apple apron arrow aspen badge bagel baker bamboo banjo barley basil basket beacon beaver
bell berry birch biscuit blanket bloom bobcat bonnet bottle boulder bramble brick bridge bronze brook
bucket buffalo bugle bundle burrow butter cabin cactus camel candle canoe canvas canyon carpet carrot
castle cedar cellar cherry chimney cider citrus clover cobalt coconut comet compass copper coral cotton
cougar cradle crane crater cricket crystal cupboard dagger daisy dolphin donkey dragon drum eagle easel
ember emerald falcon feather fennel ferret fiddle fig flannel flint forest fossil fountain fox galaxy
garlic garnet gazelle ginger giraffe glacier goblet gopher granite grape gravel guitar hammer harbor
harp hazel heron hickory honey hornet husky iceberg igloo indigo iris island ivory jackal jaguar jasmine
jelly jigsaw juniper kayak kettle kitten koala ladder lagoon lantern larch lemon lentil lettuce lily
limestone linen lizard llama lobster locket lotus magnet mango maple marble meadow melon mitten monsoon
moose mosaic muffin mustard nectar needle nickel nutmeg oasis oatmeal octopus olive onion orchid otter
owl oyster paddle panda papaya parrot peanut pebble pelican pepper piano pigeon pillow plum pocket
pony poplar potato puffin pumpkin quartz quill quilt rabbit radish raven reef ribbon river robin rocket
saddle saffron salmon sandal satchel scarf seal sesame shovel silver sparrow spinach sponge spruce
squash squid statue summit sunflower swan teapot thistle thimble thunder tiger timber toffee tomato
topaz tractor trumpet tulip tundra turnip turtle umbrella valley velvet violet vulture walnut walrus
wagon whale willow wombat yogurt zebra zinc
"""
_BAN = {'start', 'done', 'task', 'agent', 'line', 'log', 'open', 'sub', 'answer', 'delegated', 'return', 'def'}
NAMES = sorted(set(_RAW.split()) - _BAN)
for _a in NAMES:
    for _b in NAMES:
        assert _a == _b or not _b.startswith(_a), (_a, _b)
assert len(NAMES) >= 200, len(NAMES)
