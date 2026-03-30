import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TIMEZONE_FINDER = TimezoneFinder()

# ------------------------------------------------------------
# Timezone helpers
# ------------------------------------------------------------

def timezone_name_from_coordinates(latitude, longitude):
    tz_name = TIMEZONE_FINDER.timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        tz_name = TIMEZONE_FINDER.closest_timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        raise ValueError(f"Could not determine a timezone for coordinates ({latitude}, {longitude})")
    return tz_name


def timezone_from_coordinates(latitude, longitude):
    return ZoneInfo(timezone_name_from_coordinates(latitude, longitude))


# ------------------------------------------------------------
# Utility conversions
# ------------------------------------------------------------
def deg2rad(d):
    return math.radians(d)

def rad2deg(r):
    return math.degrees(r)

# ------------------------------------------------------------
# Solar position (NOAA approximation)
# Returns solar elevation and azimuth in degrees
# ------------------------------------------------------------

def print_sunrise_sunset_times(data, local_zone=None):
    if local_zone is None:
        local_zone = timezone.utc
    elif isinstance(local_zone, str):
        local_zone = ZoneInfo(local_zone)

    # When is sunset/sunrise?
    last_elev = -999
    for current, strength, elev in data:
        local_time = current.astimezone(local_zone)
        # Check for crossing the horizon (-0.833 for refraction)
        if last_elev <= -0.833 and elev > -0.833:
            print(f"Sunrise: {local_time}")
        elif last_elev >= -0.833 and elev < -0.833:
            print(f"Sunset: {local_time}")
        last_elev = elev

def solar_position(dt_utc, latitude, longitude, local_zone=None):
    # Ensure we are working with UTC for the orbital geometry
    # but using local clock time for the True Solar Time calculation
    if local_zone is None:
        local_zone = timezone_from_coordinates(latitude, longitude)
    elif isinstance(local_zone, str):
        local_zone = ZoneInfo(local_zone)
    local = dt_utc.astimezone(local_zone)
    
    # 1. Day of year
    n = local.timetuple().tm_yday
    
    # 2. Fractional year (gamma) in radians - best to use UTC here
    utc_hour = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    gamma = 2 * math.pi / 365 * (n - 1 + (utc_hour - 12) / 24)

    # 3. Equation of time (minutes) and Declination (radians)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma) 
             - 0.014615 * math.cos(2*gamma) - 0.040849 * math.sin(2*gamma))
    
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) 
           - 0.006758 * math.cos(2*gamma) + 0.000907 * math.sin(2*gamma) 
           - 0.002697 * math.cos(3*gamma) + 0.00148 * math.sin(3*gamma))

    # 4. True Solar Time (TST)
    # TST = Local Time + EqT + 4*(Lon - 15*Offset)
    local_hour = local.hour + local.minute/60 + local.second/3600
    timezone_offset = local.utcoffset().total_seconds() / 3600
    
    time_offset = eqtime + 4 * longitude - 60 * timezone_offset
    tst = local_hour * 60 + time_offset

    # 5. Hour Angle (degrees)
    ha = tst / 4 - 180
    
    # 6. Elevation
    lat_rad = math.radians(latitude)
    ha_rad = math.radians(ha)
    
    cos_zenith = (math.sin(lat_rad) * math.sin(decl) + 
                  math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad))
    
    # Clamp for safety
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = 90 - math.degrees(zenith)

    # 7. Azimuth
    sin_az = -(math.sin(ha_rad) * math.cos(decl)) / math.sin(zenith)
    cos_az = (math.sin(decl) - math.sin(lat_rad) * math.cos(zenith)) / (math.cos(lat_rad) * math.sin(zenith))
    azimuth = (math.degrees(math.atan2(sin_az, cos_az)) + 360) % 360

    return elevation, azimuth


# ------------------------------------------------------------
# Rock face illumination test
# ------------------------------------------------------------
def is_sunlit(sun_azimuth, sun_elevation, face_azimuth):
    """
    face_azimuth = direction the rock face LOOKS toward (deg)
    """

    if sun_elevation <= 0:
        return False

    # Angular difference
    diff = abs((sun_azimuth - face_azimuth + 180) % 360 - 180)

    # Sun is in front hemisphere
    return diff < 90


# ------------------------------------------------------------
# Main calculation
# ------------------------------------------------------------
def sun_shade_periods(
    date,
    latitude,
    longitude,
    face_azimuth,
    timestep_minutes=5,
):
    """
    date : datetime.date
    face_azimuth :
        0° = North
        90° = East
        180° = South
        270° = West
    """

    local_zone = timezone_from_coordinates(latitude, longitude)

    start_local = datetime(
        date.year, date.month, date.day,
        0, 0, tzinfo=local_zone
    )

    start = start_local.astimezone(timezone.utc)
    results = []

    current = start
    end = start + timedelta(days=1)

    while current < end:
        elev, az = solar_position(current, latitude, longitude, local_zone)
        strength = illumination_strength(az, elev, face_azimuth)

        results.append((current, strength, elev))
        current += timedelta(minutes=timestep_minutes)

    return results

def illumination_strength(sun_az, sun_el, face_az):
    """
    Returns continuous illumination value:
        0 = shade
        1 = maximum direct exposure
    """

    if sun_el <= 0:
        return 0.0

    # convert to radians
    az_s = math.radians(sun_az)
    el_s = math.radians(sun_el)
    az_f = math.radians(face_az)

    # Sun direction vector
    sx = math.cos(el_s) * math.sin(az_s)
    sy = math.cos(el_s) * math.cos(az_s)
    sz = math.sin(el_s)

    # Vertical wall normal vector
    nx = math.sin(az_f)
    ny = math.cos(az_f)
    nz = 0.0

    # dot product
    dot = sx*nx + sy*ny + sz*nz

    return max(0.0, dot)

def normalize(arr):
    max_val = max(x for x in arr)
    if max_val == 0:
        return arr
    return [x / max_val for x in arr]

def plot_sun_profile(data, local_zone=None):

    times = [t for t, _, _ in data]
    strength = [s for _, s, _ in data]
    elevation = [e for _, _, e in data]

    if local_zone is None:
        local_zone = timezone.utc
    elif isinstance(local_zone, str):
        local_zone = ZoneInfo(local_zone)

    times = [t.astimezone(local_zone) for t in times]
    strength = normalize(strength)
    elevation = normalize(elevation)

    plt.figure(figsize=(12,5))

    # illumination on wall
    plt.plot(times, strength, label="Wall Illumination")

    # raw solar elevation (sun peak visualization)
    plt.plot(times, elevation, linestyle="--",
             label="Solar Elevation (deg)")

    label_tz = local_zone.tzname(times[0]) if times else str(local_zone)
    plt.xlabel(f"Time ({label_tz})")
    plt.ylabel("Relative Illumination / Elevation")
    plt.title("Rock Face Solar Exposure Profile")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=local_zone))
    ax.xaxis.set_major_locator(mdates.HourLocator(tz=local_zone))

    plt.fill_between(times, strength, step="mid", alpha=0.3)

    plt.axhline(0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":

    import datetime as dt

    # Halfdome South West Face
    # latitude = 37.7414520
    # longitude = -119.5383850
    # face_azimuth = 225   # SW-facing

    # Spiliorema
    latitude = 39.112985
    longitude = 22.720292
    face_azimuth = 225   # SW-facing

    date = dt.date(2026, 3, 28)

    data = sun_shade_periods(
        date,
        latitude,
        longitude,
        face_azimuth,
        timestep_minutes=5
    )

    local_zone = timezone_from_coordinates(latitude, longitude)
    print_sunrise_sunset_times(data, local_zone)
    plot_sun_profile(data, local_zone)
