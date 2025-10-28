import rerun as rr

def log_groundplane_and_origin(entity_path: str = ""):
    # log groundplane
    rr.log(
        f"{entity_path}/groundplane",
        rr.Mesh3D(
            vertex_positions=[[500, 500, 0], [500, -500, 0], [-500, -500, 0], [500, 500, 0], [-500, -500, 0], [-500, 500, 0]],
            vertex_colors=[[94, 74, 74], [74, 94, 74], [74, 74, 94], [94, 74, 74], [74, 74, 94], [74, 94, 74]],
        ),
        static=True
    )

    # log origin
    rr.log(
        f"{entity_path}/origin",
        rr.Arrows3D(
            vectors=[[50,0,0], [0,50,0], [0,0,50]],
            origins=[[0,0,0], [0,0,0], [0,0,0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=[1,1,1]
        ),
        static=True
    )

    # log arena
    rr.log(
        f"{entity_path}/arena",
        rr.Boxes3D(
            centers=[[0, 0, 500]],
            half_sizes=[[500, 500, 500]],
            colors=[[255, 255, 255]]
        ),
        static=True
    )